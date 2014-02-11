#!/usr/bin/env python

import itertools
import collections
import traceback

import scipy.sparse
import numpy as np

from . import tile, extent
from spartan import util, core, blob_ctx, rpc
from spartan.util import Assert
from spartan import sparse
from spartan.config import FLAGS

# number of elements per tile
DEFAULT_TILE_SIZE = 100000


def take_first(a,b):
  return a


def compute_splits(shape, tile_hint=None, num_shards=-1):
  '''Split an array of shape ``shape`` into `Extent`s containing roughly `TILE_SIZE` elements.
 
  :param shape: tuple
  :param tile_hint: tuple indicating the desired tile shape 
  :rtype: list of `Extent`
  '''

  util.log_info('Splitting %s %s %s', shape, tile_hint, num_shards)

  splits = [None] * len(shape)
  if tile_hint is None:
    if num_shards != -1:
      tile_size = np.prod(shape) / num_shards
    else:
      tile_size = DEFAULT_TILE_SIZE
  
    # try to make reasonable tiles
    if len(shape) == 0:
      return { extent.create([], [], ()) :  0 }
   
    weight = 1
    
    # split each dimension into tiles.  the first dimension
    # is kept contiguous if possible.
    for dim in reversed(range(len(shape))):
      step = max(1, tile_size / weight)
      dim_splits = []
      for i in range(0, shape[dim], step):
        dim_splits.append((i, min(shape[dim], i + step)))
        
      splits[dim] = dim_splits
      weight *= shape[dim]
  else:
    Assert.eq(len(tile_hint), len(shape),
              '#dimensions in tile hint does not match shape %s vs %s' % (tile_hint, shape))
    for dim in range(len(shape)):
      dim_splits = []
      step = tile_hint[dim]
      #Assert.le(step, shape[dim])
      for i in range(0, shape[dim], step):
        dim_splits.append((i, min(shape[dim],  i + step)))
      splits[dim] = dim_splits

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
    

def _tile_mapper(blob_id, blob, array=None, user_fn=None, **kw):
  '''Invoke ``user_fn`` on ``blob``, and construct tiles from the results.'''
  ex = array.extent_for_blob(blob_id)
  return user_fn(ex, **kw)


class DistArray(object):
  '''The interface required for distributed arrays.'''

  def fetch(self, ex):
    raise NotImplementedError

  def update(self, ex, data):
    raise NotImplementedError

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

  def glom(self):
    #util.log_info('Glomming: %s', self.shape)
    return self.select(np.index_exp[:])

  def map_to_array(self, mapper_fn, kw=None):
    results = self.foreach_tile(mapper_fn=mapper_fn, kw=kw)
    extents = {}
    for blob_id, d in results.iteritems():
      for ex, id in d:
        extents[ex] = id
    return from_table(extents)
  
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

    if self.ctx.is_master():
      util.log_info('New array: %s, %s, %s tiles', shape, dtype, len(tiles))

    #util.log_info('%s', extents)
    Assert.isinstance(tiles, dict)

    self.blob_to_ex = {}
    for k,v in tiles.iteritems():
      Assert.isinstance(k, extent.TileExtent)
      Assert.isinstance(v, core.BlobId)
      self.blob_to_ex[v] = k
      #util.log_info('Blob: %s', v)

    self.tiles = tiles
    self.id = ID_COUNTER.next()

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
    if self.ctx.worker_id == blob_ctx.MASTER_ID:
      #util.log_info('Destroying table... %s', self.id)
      tiles = self.tiles.values()
      _pending_destructors.extend(tiles)

  def id(self):
    return self.table.id()

  def extent_for_blob(self, id):
    return self.blob_to_ex[id]
  
  def tile_shape(self):
    scounts = collections.defaultdict(int)
    for ex in self.tiles.iterkeys():
      scounts[ex.shape] += 1
    
    return sorted(scounts.items(), key=lambda kv: kv[1])[-1][0]

  def foreach_tile(self, mapper_fn, kw=None):
    ctx = blob_ctx.get()

    if kw is None: kw = {}
    kw['array'] = self
    kw['user_fn'] = mapper_fn

    return ctx.map(self.tiles.values(),
                   mapper_fn = _tile_mapper,
                   reduce_fn = None,
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
      blob_id = self.tiles[region]
      tgt = ctx.get(blob_id, extent.offset_slice(ex, intersection))
      return tgt
    
    #util.log_warn('Remote fetch.')
    splits = list(extent.find_overlapping(self.tiles.iterkeys(), region))

    #util.log_info('Target shape: %s, %d splits', region.shape, len(splits))
    #util.log_info('Fetching %d tiles', len(splits))

    futures = []
    for ex, intersection in splits:
      blob_id = self.tiles[ex]
      futures.append(ctx.get(blob_id, extent.offset_slice(ex, intersection), wait=False))
    
    # stitch results back together
    # if we have any masked tiles, then we need to create a masked array.
    # otherwise, create a dense array.
    results = [r.data for r in rpc.wait_for_all(futures)]
   
    DENSE = 0
    MASKED = 1
    SPARSE = 2
    
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
      tgt = scipy.sparse.lil_matrix(region.shape, dtype=self.dtype)
    else:
      tgt = np.ndarray(region.shape, dtype=self.dtype)
    
    for (ex, intersection), result in zip(splits, results):
      dst_slice = extent.offset_slice(region, intersection)
      #util.log_info('ex:%s region:%s intersection:%s dst_slice:%s result:%s', ex, region, intersection, dst_slice, result)
      #util.log_info('tgt.shape:%s result.shape:%s tgt.type:%s result.type:%s', tgt[dst_slice].shape, result.shape, type(tgt), type(result))
      if np.all(result.shape):
        tgt[dst_slice] = result

    return tgt
    #return tile.data[]
   
  def update_slice(self, slc, data):
    return self.update(extent.from_slice(slc, self.shape), data)
     
  def update(self, region, data, wait=True):
    ctx = blob_ctx.get()
    Assert.isinstance(region, extent.TileExtent)
    Assert.eq(region.shape, data.shape,
              'Size of extent does not match size of data')

    # exact match
    if region in self.tiles:
      blob_id = self.tiles[region]
      dst_slice = extent.offset_slice(region, region)
      #util.log_info('EXACT: %s %s ', region, dst_slice)
      return ctx.update(blob_id, dst_slice, data, self.reducer_fn, wait=wait)
    
    splits = list(extent.find_overlapping(self.tiles, region))
    futures = []
    slices = []
    #util.log_info('%s: Updating %s tiles with data:%s', region, len(splits), data)
    
    for dst_extent, intersection in splits:
      #util.log_info('%s %s %s', region, dst_extent, intersection)

      blob_id = self.tiles[dst_extent]

      src_slice = extent.offset_slice(region, intersection)
      dst_slice = extent.offset_slice(dst_extent, intersection)
   
      shape = [slice.stop - slice.start for slice in dst_slice]
      if np.all(shape):   
        slices.append((blob_id, src_slice, dst_slice))
      #util.log_info('Update src:%s dst:%s data shape:%s', src_slice, dst_slice, data.shape)
    
    slices.sort(key=lambda x: x[1][0].start)
    #util.log_info("Update: slices:%s", slices)
    result = sparse.multiple_slice(data, slices)
    
    for (blob_id, dst_slice, update_data) in result:
        #update_data = sparse.slice(data, src_slice)       
        #if update_data is not None:
      #update_tile = tile.from_intersection(dst_key, intersection, data[src_slice])
      #util.log_info('%s %s %s %s', dst_key.shape, intersection.shape, blob_id, update_tile)
      #util.log_info("Updating %d tile %s with dst_ex %s intersection %s with data %s slice %s", len(splits), blob_id, dst_extent, intersection, data.nonzero()[0], data[src_slice].nonzero()[0])
      futures.append(ctx.update(blob_id, 
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

  extents = compute_splits(shape, tile_hint, ctx.num_workers * 4)
  tiles = {}
  tile_type = tile.TYPE_SPARSE if sparse else tile.TYPE_DENSE
  
  if FLAGS.tile_assignment_strategy == 'round_robin':
    for ex, i in extents.iteritems():    
      tiles[ex] = ctx.create(
                    tile.from_shape(ex.shape, dtype, tile_type=tile_type), 
                    hint=i)
  elif FLAGS.tile_assignment_strategy == 'performance':
    worker_scores = ctx.get_worker_scores()
    for ex, i in extents.iteritems():    
      tiles[ex] = ctx.create(
                  tile.from_shape(ex.shape, dtype, tile_type=tile_type), 
                  hint=worker_scores[i%len(worker_scores)][0])
  else: #random
    for ex in extents:
      tiles[ex] = ctx.create(tile.from_shape(ex.shape, dtype, tile_type=tile_type))
      
  for ex in extents:
    tiles[ex] = tiles[ex].wait().blob_id

  #for ex, i in extents.iteritems():
  #  util.log_warn("i:%d ex:%s, blob_id:%s", i, ex, tiles[ex])
    
  array = DistArrayImpl(shape=shape, dtype=dtype, tiles=tiles, reducer_fn=reducer, sparse=sparse)
  
  for blob_id in tiles.values():
    ctx.register_blob(blob_id, array)
    
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
  for ex, blob_id in X.tiles.iteritems():
    if blob_id.worker not in worker_to_tiles:
      worker_to_tiles[blob_id.worker] = [ex]
    else:
      worker_to_tiles[blob_id.worker].append(ex)
  
  for worker_id, ex_list in worker_to_tiles.iteritems():
    for ex in ex_list:
      tiles[ex] = ctx.create(
                  tile.from_shape(ex.shape, dtype, tile_type=tile_type), 
                  hint=worker_id+1)
      
  for ex in tiles:
    tiles[ex] = tiles[ex].wait().blob_id

  #for ex, i in extents.iteritems():
  #  util.log_warn("i:%d ex:%s, blob_id:%s", i, ex, tiles[ex])
    
  array = DistArrayImpl(shape=shape, dtype=dtype, tiles=tiles, reducer_fn=reducer, sparse=sparse)
  
  for blob_id in tiles.values():
    ctx.register_blob(blob_id, array)
    
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
    key, blob_id = extents.iteritems().next()
    util.log_info('%s :: %s', key, blob_id)
    
    #dtype = blob_ctx.get().run_on_tile(blob_id, lambda t: t.dtype).wait()
    dtype, sparse = blob_ctx.get().tile_op(blob_id, lambda t: (t.dtype, t.type == tile.TYPE_SPARSE)).result
    #dtype = None
  else:
    # empty table; default dtype.
    dtype = np.float
    sparse = False
  
  array = DistArrayImpl(shape=shape, dtype=dtype, tiles=extents, reducer_fn=None, sparse=sparse)
  
  ctx = blob_ctx.get()
  for blob_id in extents.values():
    ctx.register_blob(blob_id, array)
  
  return array

class LocalWrapper(DistArray):
  '''
  Provide the `DistArray` interface for local data.
  '''
  def __init__(self, data):
    self._data = np.asarray(data)
    self.bad_tiles = []
    #assert not isinstance(data, core.BlobId)
    Assert.isinstance(data, (np.ndarray, int, float))
    #print 'Wrapping: %s %s (%s)' % (data, type(data), np.isscalar(data))
    #print 'DATA: %s' % type(self._data)

  @property
  def dtype(self):
    return self._data.dtype

  @property
  def shape(self):
    return self._data.shape

  def fetch(self, ex):
    return self._data[ex.to_slice()]

  def map_to_array(self, mapper_fn, kw=None):
    return self.foreach_tile(mapper_fn=mapper_fn, kw=kw)

  def foreach_tile(self, mapper_fn, kw=None):
    #print 'Mapping: ', mapper_fn, ' over ', self._data
    if kw is None: kw = {}
    ex = extent.from_slice(np.index_exp[:], self.shape)
    map_result = mapper_fn(ex, **kw)
    result = map_result.result
    
    assert len(result) == 1
    result_ex, tile_id = result[0]
    
    Assert.isinstance(tile_id, core.BlobId)
    ctx = blob_ctx.get()
    
    result_data = ctx.get(tile_id, slice(None, None, None))
    return as_array(result_data)


def as_array(data):
  if isinstance(data, DistArray):
    return data

  # TODO(power) -- promote numpy arrays to distarrays?
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
  


def _slice_mapper(ex, **kw):
  '''
  Run when mapping over a slice.
  Computes the intersection of the current tile and a global slice.
  If the slice is non-zero, then run the user mapper function.
  Otherwise, do nothing.
  
  :param ex:
  :param tile: 
  :param mapper_fn: User mapper function.
  :param slice: `TileExtent` representing the slice of the input array.
  '''

  mapper_fn = kw['_slice_fn']
  slice_extent = kw['_slice_extent']

  fn_kw = kw['fn_kw']
  if fn_kw is None: fn_kw = {}

  intersection = extent.intersection(slice_extent, ex)
  if intersection is None:
    from spartan.expr.map import MapResult
    return MapResult([], None)

  offset = extent.offset_from(slice_extent, intersection)
  offset.array_shape = slice_extent.shape

  subslice = extent.offset_slice(ex, intersection)

  result = mapper_fn(offset, **fn_kw)
  #util.log_info('Slice mapper[%s] %s %s -> %s', mapper_fn, offset, subslice, result)
  return result

class Slice(DistArray):
  def __init__(self, darray, idx):
    if not isinstance(idx, extent.TileExtent):
      idx = extent.from_slice(idx, darray.shape)
    util.log_info('New slice: %s', idx)
    
    Assert.isinstance(darray, DistArray)
    self.darray = darray
    self.slice = idx
    self.shape = self.slice.shape
    intersections = [extent.intersection(self.slice, ex) for ex in self.darray.tiles]
    intersections = [ex for ex in intersections if ex is not None]
    offsets = [extent.offset_from(self.slice, ex) for ex in intersections]
    self.tiles = offsets
    self.dtype = darray.dtype
  
  @property
  def bad_tiles(self):
    bad_intersections = [extent.intersection(self.slice, ex) for ex in self.darray.bad_tiles]
    return [ex for ex in bad_intersections if ex is not None]
    
  def foreach_tile(self, mapper_fn, kw):
    return self.darray.foreach_tile(mapper_fn = _slice_mapper,
                                    kw={'fn_kw' : kw,
                                        '_slice_extent' : self.slice,
                                        '_slice_fn' : mapper_fn })

  def fetch(self, idx):
    offset = extent.compute_slice(self.slice, idx.to_slice())
    return self.darray.fetch(offset)


def broadcast_mapper(ex, tile, mapper_fn=None, bcast_obj=None):
  raise NotImplementedError

class Broadcast(DistArray):
  '''A broadcast object mimics the behavior of Numpy broadcasting.
  
  Takes an input of shape (x, y) and a desired output shape (x, y, z),
  the broadcast object reports shape=(x,y,z) and overrides __getitem__
  to return the appropriate values.
  '''
  def __init__(self, base, shape):
    Assert.isinstance(base, (np.ndarray, DistArray))
    Assert.isinstance(shape, tuple)
    self.base = base
    self.shape = shape
    self.dtype = base.dtype
    self.bad_tiles = []

    
  def __repr__(self):
    return 'Broadcast(%s -> %s)' % (self.base, self.shape)
 
  def fetch(self, ex):
    # make a template to pass to numpy broadcasting
    template = np.ndarray(ex.shape, dtype=self.base.dtype)
   
    # convert the extent to the base form

    # first drop extra dimensions
    while len(ex.shape) > len(self.base.shape):
      ex = extent.drop_axis(ex, 0)
      
    # fold down expanded dimensions
    ul = []
    lr = []
    for i in xrange(len(self.base.shape)):
      size = self.base.shape[i]
      if size == 1:
        ul.append(0)
        lr.append(1)
      else:
        ul.append(ex.ul[i])
        lr.append(ex.lr[i])
  
    ex = extent.create(ul, lr, self.base.shape) 
    fetched = self.base.fetch(ex)
    
    _, bcast = np.broadcast_arrays(template, fetched)
    
    util.log_debug('bcast: %s %s', fetched.shape, template.shape)
    return bcast 


def broadcast(args):
  '''Convert the list of arrays in ``args`` to have the same shape.
  
  Extra dimensions are added as necessary, and dimensions of size
  1 are repeated to match the size of other arrays.
  
  :param args: List of `DistArray`
  '''
  
  if len(args) == 1:
    return args

  orig_shapes = [list(x.shape) for x in args]
  dims = [len(shape) for shape in orig_shapes]
  max_dim = max(dims)
  new_shapes = []
  
  # prepend filler dimensions for smaller arrays
  for i in range(len(orig_shapes)):
    diff = max_dim - len(orig_shapes[i])
    new_shapes.append([1] * diff + orig_shapes[i])
 
  # check shapes are valid
  # for each axis, all arrays should either share the 
  # same size, or have size == 1
  for axis in range(max_dim):
    axis_shape = set(shp[axis] for shp in new_shapes)
   
    assert len(axis_shape) <= 2, 'Mismatched shapes for broadcast: %s' % orig_shapes
    if len(axis_shape) == 2:
      assert 1 in axis_shape, 'Mismatched shapes for broadcast: %s' % orig_shapes
  
    # now lift the inputs with size(axis) == 1 
    # to have the maximum size for the axis 
    max_size = max(shp[axis] for shp in new_shapes)
    for shp in new_shapes:
      shp[axis] = max_size
    
  # wrap arguments with missing dims in a Broadcast object.
  results = []
  for i in range(len(args)):
    if new_shapes[i] == orig_shapes[i]:
      results.append(args[i])
    else:
      results.append(Broadcast(args[i], tuple(new_shapes[i])))
    
  #util.log_debug('Broadcast result: %s', results)
  return results

def _size(v):
  if isinstance(v, Broadcast):
    return (0, np.prod(v.base.shape)) 
  return (1, np.prod(v.shape))

def largest_value(vals):
  '''
  Return the largest array (using the underlying size for Broadcast objects).
  
  :param vals: List of `DistArray`. 
  '''
  return max(vals, key=_size)

