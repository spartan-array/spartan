#!/usr/bin/env python

import itertools
import collections
import traceback

import appdirs
import scipy.sparse
import numpy as np

from . import tile, extent, sparse
from .. import util, core, blob_ctx, rpc
from ..core import LocalKernelResult
from ..util import Assert
from ..config import FLAGS
from .. import master

# number of elements per tile
DEFAULT_TILE_SIZE = 100000


def take_first(a, b):
  return a


def good_tile_shape(shape, num_shards=-1):
  ''' Compute a tile_shape (tile_hint) for the array.

  Args:
    shape: tuple. the array's shape.

  Returns:
    list: tile_shape for the array
  '''
  if num_shards != -1:
    tile_size = np.prod(shape) / num_shards
  else:
    tile_size = DEFAULT_TILE_SIZE

  # fill up dimensions...
  tile_shape = [1] * len(shape)
  idx = len(shape) - 1
  while tile_size > 1:
    tile_shape[idx] = min(shape[idx], tile_size)
    tile_size /= shape[idx]
    idx -= 1

  return tile_shape


def compute_splits(shape, tile_hint):
  ''' Based on tile_hint to compute splits for each dimension of the array of shape ``shape``

  Args:
    shape: tuple. the array's shape.
    tile_hint: tuple indicating the desired tile shape.

  Returns:
    list: splits for each dimension.
  '''
  splits = [None] * len(shape)
  for dim in range(len(shape)):
    dim_splits = []
    step = tile_hint[dim]
    #Assert.le(step, shape[dim])
    for i in range(0, shape[dim], step):
      dim_splits.append((i, min(shape[dim],  i + step)))
    splits[dim] = dim_splits

  return splits


def compute_extents(shape, tile_hint=None, num_shards=-1):
  '''Split an array of shape ``shape`` into `Extent`s. Each extent contains roughly `TILE_SIZE` elements if num_shards is -1.

  Args:
    shape: tuple. the array's shape.
    tile_hint: tuple indicating the desired tile shape.

  Returns:
    list: list of `Extent`
  '''

  util.log_debug('Splitting %s %s %s', shape, tile_hint, num_shards)

  # try to make reasonable tiles
  if len(shape) == 0:
    return {extent.create([], [], ()):  0}

  if tile_hint is None:
    tile_hint = good_tile_shape(shape, num_shards)
  else:
    Assert.eq(len(tile_hint), len(shape),
              '#dimensions in tile hint does not match shape %s vs %s' %
              (tile_hint, shape))

  splits = compute_splits(shape, tile_hint)

  result = {}
  idx = 0
  for slc in itertools.product(*splits):
    if num_shards != -1:
      idx = idx % num_shards

    ul, lr = zip(*slc)
    ex = extent.create(ul, lr, shape)
    result[ex] = idx
    idx += 1

  return result


def _tile_mapper(tile_id, blob, array=None, user_fn=None, **kw):
  '''Invoke ``user_fn`` on ``blob``, and construct tiles from the results.'''
  ex = array.extent_for_blob(tile_id)
  return user_fn(ex, **kw)


class DistArray(object):
  '''The interface required for distributed arrays.

  A distributed array should support:

     * ``fetch(ex)`` to fetch data
     * ``update(ex, data)`` to combine an update with existing data
     * ``foreach_tile(fn, kw)``
  '''

  def fetch(self, ex):
    '''Fetch the region specified by extent from this array.

    Args:
      ex (Extent): Region to fetch

    Returns:
      np.ndarray: Data from region.

    '''
    raise NotImplementedError

  def update(self, ex, data):
    raise NotImplementedError

  def foreach_tile(self, mapper_fn, kw):
    raise NotImplementedError

  def extent_for_blob(self, id):
    raise NotImplementedError

  def real_size(self):
    '''The actual number of elements contained by this array.

    Broadcast objects "pretend" to have a larger size than they actually do.
    For mapping across data, we want to ignore this.
    '''
    return np.prod(self.shape)

  def __len__(self):
    ''' Alias of real_size(self). '''
    return self.shape[0]

  def __repr__(self):
    return '%s(id=%s, shape=%s, dtype=%s)' % (self.__class__.__name__, id(self), self.shape, self.dtype)

  def __setitem__(self, idx, value):
    if np.isscalar(idx):
      result = self.select(slice(idx, idx + 1))
      return result[0]

    ex = extent.from_slice(idx, self.shape)
    if not isinstance(value, np.ndarray):
      a_value = np.ndarray(ex.shape, dtype=self.dtype)
      a_value[:] = value
      value = a_value
    self.update(ex, value)

  def select(self, idx):
    '''
    Effectively __getitem__.

    Renamed to avoid the chance of accidentally using a slow, local operation on
    a distributed array.
    '''
    if isinstance(idx, extent.TileExtent):
      return self.fetch(idx)

    if np.isscalar(idx):
      result = self.select(slice(idx, idx + 1))
      return result[0]

    ex = extent.from_slice(idx, self.shape)
    #util.log_info('Select: %s + %s -> %s', idx, self.shape, ex)
    return self.fetch(ex)

  def __getitem__(self, idx):
    return self.select(idx)

  def glom(self):
    #util.log_info('Glomming: %s', self.shape)
    return self.select(np.index_exp[:])

  def map_to_array(self, mapper_fn, kw=None):
    results = self.foreach_tile(mapper_fn=mapper_fn, kw=kw)
    extents = {}
    for tile_id, d in results.iteritems():
      for ex, id in d:
        extents[ex] = id
    return from_table(extents)

  def __hash__(self):
    return id(self)

  @property
  def ndim(self):
    return len(self.shape)

ID_COUNTER = iter(xrange(10000000))

# List of tiles to be destroyed at the next safe point.
_pending_destructors = []


class DistArrayImpl(DistArray):
  def __init__(self, shape, dtype, tiles, reducer_fn, sparse):
    #traceback.print_stack()
    self.shape = shape
    self.dtype = dtype
    self.reducer_fn = reducer_fn
    self.sparse = sparse
    self.bad_tiles = []
    self.ctx = blob_ctx.get()

    Assert.not_null(dtype)

    Assert.isinstance(tiles, dict)

    self.blob_to_ex = {}
    for k, v in tiles.iteritems():
      Assert.isinstance(k, extent.TileExtent)
      Assert.isinstance(v, core.TileId)
      self.blob_to_ex[v] = k
      #util.log_info('Blob: %s', v)

    self.tiles = tiles
    self.id = ID_COUNTER.next()

    if self.ctx.is_master():
      #util.log_info('New array: %s, %s, %s tiles', shape, dtype, len(tiles))
      if _pending_destructors:
        self.ctx.destroy_all(_pending_destructors)
        del _pending_destructors[:]

  def __reduce__(self):
    return (DistArrayImpl, (self.shape, self.dtype, self.tiles, self.reducer_fn, self.sparse))

  def __del__(self):
    '''Destroy this array.

    NB: Destruction is actually deferred until the next usage of the
    blob_ctx.  __del__ can be called at anytime, including the
    invocation of a RPC call, which leads to odd/bad behavior.
    '''
    if self.ctx.is_master():
      # Logging during shutdown doesn't work.
      #util.log_debug('Destroying table... %s', self.id)
      tiles = self.tiles.values()
      if isinstance(_pending_destructors, list):
        _pending_destructors.extend(tiles)

  def id(self):
    return self.table.id()

  def extent_for_blob(self, id):
    return self.blob_to_ex[id]

  def tile_shape(self):
    scounts = collections.defaultdict(int)
    for ex in self.tiles.iterkeys():
      scounts[ex.shape] += 1

    return sorted(scounts.items(), key=lambda kv: (kv[1], kv[0]))[-1][0]

  def foreach_tile(self, mapper_fn, kw=None):
    ctx = blob_ctx.get()

    if kw is None: kw = {}
    kw['array'] = self
    kw['user_fn'] = mapper_fn

    return ctx.map(self.tiles.values(),
                   mapper_fn=_tile_mapper,
                   kw=kw)

  def fetch(self, region):
    '''
    Return a local numpy array for the given region.

    If necessary, data will be copied from remote hosts to fill the region.
    :param region: `Extent` indicating the region to fetch.
    '''
    Assert.isinstance(region, extent.TileExtent)
    Assert.eq(region.array_shape, self.shape)
    Assert.eq(len(region.ul), len(self.shape))
    assert np.all(region.lr <= self.shape), 'Requested region is out of bounds: %s > %s' % (region, self.shape)
    #util.log_info('FETCH: %s %s', self.shape, region)

    ctx = blob_ctx.get()

    # special case exact match against a tile
    if region in self.tiles:
      #util.log_warn('Exact match.')
      ex, intersection = region, region
      tile_id = self.tiles[region]
      tgt = ctx.get(tile_id, extent.offset_slice(ex, intersection))
      return tgt

    #util.log_warn('Remote fetch.')
    splits = list(extent.find_overlapping(self.tiles.iterkeys(), region))

    #util.log_info('Target shape: %s, %d splits', region.shape, len(splits))
    #util.log_info('Fetching %d tiles', len(splits))

    futures = []
    for ex, intersection in splits:
      tile_id = self.tiles[ex]
      futures.append(ctx.get(tile_id, extent.offset_slice(ex, intersection), wait=False))

    # stitch results back together
    # if we have any masked tiles, then we need to create a masked array.
    # otherwise, create a dense array.
    results = [r.data for r in rpc.wait_for_all(futures)]

    DENSE = 0
    MASKED = 1
    SPARSE = 2

    # If there is only one slice, no need to do copy
    if len(splits) == 1:
      return results[0]

    output_type = DENSE
    for r in results:
      if isinstance(r, np.ma.MaskedArray) and output_type == DENSE:
        output_type = MASKED
      if scipy.sparse.issparse(r):
        output_type = SPARSE

    if output_type == MASKED:
      tgt = np.ma.MaskedArray(np.ndarray(region.shape, dtype=self.dtype))
      tgt.mask = 0
    elif output_type == SPARSE:
      tgt = scipy.sparse.coo_matrix(region.shape, dtype=self.dtype)
      tgt = sparse.convert_sparse_array(tgt)
    else:
      tgt = np.ndarray(region.shape, dtype=self.dtype)

    for (ex, intersection), result in zip(splits, results):
      dst_slice = extent.offset_slice(region, intersection)
      #util.log_info('ex:%s region:%s intersection:%s dst_slice:%s result:%s', ex, region, intersection, dst_slice, result)
      #util.log_info('tgt.shape:%s result.shape:%s tgt.type:%s result.type:%s', tgt[dst_slice].shape, result.shape, type(tgt), type(result))
      if extent.all_nonzero_shape(result.shape):
        if output_type == SPARSE:
          tgt = sparse.compute_sparse_update(tgt, result, dst_slice)
        else:
          tgt[dst_slice] = result

    return tgt

  def update_slice(self, slc, data):
    return self.update(extent.from_slice(slc, self.shape), data)

  def update(self, region, data, wait=True):
    ctx = blob_ctx.get()
    Assert.isinstance(region, extent.TileExtent)
    Assert.eq(region.shape, data.shape,
              'Size of extent does not match size of data')

    # exact match
    if region in self.tiles:
      tile_id = self.tiles[region]
      dst_slice = extent.offset_slice(region, region)
      #util.log_info('EXACT: %s %s ', region, dst_slice)
      return ctx.update(tile_id, dst_slice, data, self.reducer_fn, wait=wait)

    futures = []
    slices = []

    if region.shape == self.shape:
      for ex, tile_id in self.tiles.iteritems():
        slices.append((tile_id, ex.to_slice(), extent.offset_slice(ex, ex)))
    else:
      splits = list(extent.find_overlapping(self.tiles, region))
      #util.log_info('%s: Updating %s tiles with data:%s', region, len(splits), data)

      for dst_extent, intersection in splits:
        #util.log_info('%s %s %s', region, dst_extent, intersection)

        tile_id = self.tiles[dst_extent]

        src_slice = extent.offset_slice(region, intersection)
        dst_slice = extent.offset_slice(dst_extent, intersection)

        shape = [slice.stop - slice.start for slice in dst_slice]
        if extent.all_nonzero_shape(shape):
          slices.append((tile_id, src_slice, dst_slice))
        #util.log_info('Update src:%s dst:%s data shape:%s', src_slice, dst_slice, data.shape)

    slices.sort(key=lambda x: x[1][0].start)
    #util.log_info("Update: slices:%s", slices)
    result = sparse.multiple_slice(data, slices)

    for (tile_id, dst_slice, update_data) in result:
      futures.append(ctx.update(tile_id,
                                dst_slice,
                                update_data,
                                self.reducer_fn,
                                wait=False))

    if wait:
      rpc.wait_for_all(futures)
    else:
      return rpc.FutureGroup(futures)


def create(shape,
           dtype=np.float,
           sharder=None,
           reducer=None,
           tile_hint=None,
           sparse=False):
  '''Make a new, empty DistArray'''
  ctx = blob_ctx.get()
  dtype = np.dtype(dtype)
  shape = tuple(shape)

  util.log_debug('Creating a new distarray with shape %s', str(shape))
  extents = compute_extents(shape, tile_hint, ctx.num_workers)
  tiles = {}
  tile_type = tile.TYPE_SPARSE if sparse else tile.TYPE_DENSE

  if FLAGS.tile_assignment_strategy == 'round_robin':
    for ex, i in extents.iteritems():
      tiles[ex] = ctx.create(
                    tile.from_shape(ex.shape, dtype, tile_type=tile_type),
                    hint=i)
  elif FLAGS.tile_assignment_strategy == 'performance':
    worker_scores = master.get().get_worker_scores()
    for ex, i in extents.iteritems():
      tiles[ex] = ctx.create(
                  tile.from_shape(ex.shape, dtype, tile_type=tile_type),
                  hint=worker_scores[i % len(worker_scores)][0])
  elif FLAGS.tile_assignment_strategy == 'serpentine':
    for ex, i in extents.iteritems():
      j = i % ctx.num_workers
      if (i / ctx.num_workers) % 2 == 1:
        j = (ctx.num_workers - 1 - j)

      tiles[ex] = ctx.create(
                    tile.from_shape(ex.shape, dtype, tile_type=tile_type),
                    hint=j)
  elif FLAGS.tile_assignment_strategy == 'static':
    all_extents = list(extents.iterkeys())
    all_extents.sort()
    if hasattr(appdirs, 'user_config_dir'):  # user_config_dir new to v1.3.0
      map_file = appdirs.user_config_dir('spartan') + '/tiles_map'
    else:
      map_file = appdirs.user_data_dir('spartan') + '/tiles_map'
    with open(map_file) as fp:
      for ex in all_extents:
        worker = int(fp.readline().strip())
        tiles[ex] = ctx.create(
                    tile.from_shape(ex.shape, dtype, tile_type=tile_type),
                    hint=worker)
  else: #  random
    for ex in extents:
      tiles[ex] = ctx.create(tile.from_shape(ex.shape, dtype, tile_type=tile_type))

  for ex in extents:
    tiles[ex] = tiles[ex].wait().tile_id

  #for ex, i in extents.iteritems():
  #  util.log_warn("i:%d ex:%s, tile_id:%s", i, ex, tiles[ex])

  array = DistArrayImpl(shape=shape, dtype=dtype, tiles=tiles, reducer_fn=reducer, sparse=sparse)
  master.get().register_array(array)
  util.log_debug('Succcessfully created a new distarray')
  return array


def from_replica(X):
  '''Make a new, empty DistArray from X'''
  ctx = blob_ctx.get()
  dtype = X.dtype
  shape = X.shape
  reducer = X.reducer_fn
  sparse = X.sparse
  tile_type = tile.TYPE_SPARSE if sparse else tile.TYPE_DENSE
  tiles = {}
  worker_to_tiles = {}
  for ex, tile_id in X.tiles.iteritems():
    if tile_id.worker not in worker_to_tiles:
      worker_to_tiles[tile_id.worker] = [ex]
    else:
      worker_to_tiles[tile_id.worker].append(ex)

  for worker_id, ex_list in worker_to_tiles.iteritems():
    for ex in ex_list:
      tiles[ex] = ctx.create(tile.from_shape(ex.shape, dtype, tile_type=tile_type),
                             hint=worker_id+1)

  for ex in tiles:
    tiles[ex] = tiles[ex].wait().tile_id

  array = DistArrayImpl(shape=shape, dtype=dtype, tiles=tiles, reducer_fn=reducer, sparse=sparse)
  master.get().register_array(array)
  return array


def from_table(extents):
  '''
  Construct a distarray from an existing table.
  Keys must be of type `Extent`, values of type `Tile`.

  Shape is computed as the maximum range of all extents.

  Dtype is taken from the dtype of the tiles.

  :param table:
  '''
  Assert.no_duplicates(extents)

  if not extents:
    shape = tuple()
  else:
    shape = extent.find_shape(extents.keys())

  if len(extents) > 0:
    # fetch one tile from the table to figure out the dtype
    key, tile_id = extents.iteritems().next()
    util.log_debug('%s :: %s', key, tile_id)

    dtype, sparse = blob_ctx.get().tile_op(tile_id, lambda t: (t.dtype, t.type == tile.TYPE_SPARSE)).result
  else:
    # empty table; default dtype.
    dtype = np.float
    sparse = False

  array = DistArrayImpl(shape=shape, dtype=dtype, tiles=extents, reducer_fn=None, sparse=sparse)
  master.get().register_array(array)
  return array


class LocalWrapper(DistArray):
  '''
  Provide the `DistArray` interface for local data.
  '''
  def __init__(self, data):
    self._data = np.asarray(data)
    self.sparse = False
    self.bad_tiles = []
    self._ex = extent.from_slice(np.index_exp[:], self.shape)
    Assert.isinstance(data, (np.ndarray, int, long, float))
    #print 'Wrapping: %s %s (%s)' % (data, type(data), np.isscalar(data))
    #print 'DATA: %s' % type(self._data)

  @property
  def dtype(self):
    return self._data.dtype

  @property
  def shape(self):
    return self._data.shape

  @property
  def tiles(self):
    # LocalWrapper doesn't actually have tiles, so return a fake tile
    # representing the entire array
    return {self._ex: core.TileId(-1, 0)}

  def extent_for_blob(self, tile_id):
    return self._ex

  def fetch(self, ex):
    return self._data[ex.to_slice()]

  def map_to_array(self, mapper_fn, kw=None):
    return self.foreach_tile(mapper_fn=mapper_fn, kw=kw)

  def foreach_tile(self, mapper_fn, kw=None):
    #print 'Mapping: ', mapper_fn, ' over ', self._data
    if kw is None: kw = {}
    map_result = mapper_fn(self._ex, **kw)
    result = map_result.result

    assert len(result) == 1
    result_ex, tile_id = result[0]

    Assert.isinstance(tile_id, core.TileId)
    ctx = blob_ctx.get()

    result_data = ctx.get(tile_id, slice(None, None, None))
    return as_array(result_data)


def as_array(data):
  '''
  Convert ``data`` to behave like a `DistArray`.

  If ``data`` is already a `DistArray`, it is returned unchanged.
  Otherwise, ``data`` is wrapped to have a `DistArray` interface.

  :param data: An input array or array-like value.
  '''
  if isinstance(data, DistArray):
    return data

  return LocalWrapper(data)


def best_locality(array, ex):
  '''
  Return the table shard with the best locality for extent `ex`.
  :param table:
  :param ex:
  '''
  splits = extent.find_overlapping(array.extents, ex)
  counts = collections.defaultdict(int)
  for key, overlap in splits:
    shard = array.extents[key]
    counts[shard] += overlap.size

  s_counts = sorted(counts.items(), key=lambda kv: kv[1])
  return s_counts[-1][0]


def largest_value(vals):
  '''
  Return the largest array (using the underlying size for Broadcast objects).

  :param vals: List of `DistArray`.
  '''
  return max(vals, key=lambda v: v.real_size())
