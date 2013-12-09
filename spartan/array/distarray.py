#!/usr/bin/env python

import itertools
import collections

import numpy as np

from . import tile, extent
from spartan import util, core, blob_ctx, rpc
from spartan.util import Assert


# number of elements per tile
TILE_SIZE = 100000


def take_first(a,b):
  return a


def compute_splits(shape, tile_hint=None, num_shards=-1):
  '''Split an array of shape ``shape`` into `Extent`s containing roughly `TILE_SIZE` elements.
 
  :param shape: tuple
  :param tile_hint: tuple indicating the desired tile shape 
  :rtype: list of `Extent`
  '''
  
  splits = [None] * len(shape)
  if tile_hint is None:
    # try to make reasonable tiles
    if len(shape) == 0:
      return { extent.create([], [], ()) :  0 }
   
    weight = 1
    
    # split each dimension into tiles.  the first dimension
    # is kept contiguous if possible.
    for dim in reversed(range(len(shape))):
      step = max(1, TILE_SIZE / weight)
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
    

def _array_mapper(blob_id, blob, array=None, user_fn=None, **kw):
  ctx = blob_ctx.get()
  ex = array.extent_for_blob(blob_id)
  results = []
  user_results = user_fn(ex, **kw)
  assert user_results is not None, user_fn

  for target_ex, target_data in user_results:
    Assert.eq(target_ex.shape, target_data.shape,
              'Bad extent for result: %s %s' % (user_fn, ex))
    blob_id = ctx.create(tile.from_data(target_data)).wait().blob_id
    results.append((target_ex, blob_id))
  return results

class DistArray(object):
  '''The interface required for distributed arrays.'''

  def fetch(self, ex):
    raise NotImplementedError

  def update(self, ex, data):
    raise NotImplementedError

  def __repr__(self):
    return '%s(shape=%s, dtype=%s)' % (self.__class__.__name__, self.shape, self.dtype)

  def select(self, idx):
    '''
    Effectively __getitem__.

    Renamed to avoid the chance of accidentally using a slow, local operation on
    a distributed array.
    '''
    if isinstance(idx, extent.TileExtent):
      return self.fetch(idx)

    if np.isscalar(idx):
      return self[idx:idx+1][0]

    ex = extent.from_slice(idx, self.shape)
    #util.log_info('Select: %s + %s -> %s', idx, self.shape, ex)
    return self.fetch(ex)

  def glom(self):
    #util.log_info('Glomming: %s', self.shape)
    return self.select(np.index_exp[:])

  def map_to_array(self, mapper_fn, kw=None):
    results = self.map_to_table(mapper_fn=mapper_fn, kw=kw)
    extents = {}
    for blob_id, d in results.iteritems():
      for ex, id in d:
        extents[ex] = id

    return from_table(extents)

  def foreach(self, mapper_fn, kw):
    return self.map_to_table(mapper_fn=mapper_fn, kw=kw)

ID_COUNTER = iter(xrange(10000000))

class DistArrayImpl(DistArray):
  def __init__(self, shape, dtype, tiles, reducer_fn):
    self.shape = shape
    self.dtype = dtype
    self.reducer_fn = reducer_fn
    self.ctx = blob_ctx.get()

    #util.log_info('%s', extents)
    Assert.isinstance(tiles, dict)

    self.blob_to_ex = {}
    for k,v in tiles.iteritems():
      Assert.isinstance(k, extent.TileExtent)
      Assert.isinstance(v, core.BlobId)
      self.blob_to_ex[v] = k

    self.tiles = tiles
    self.id = ID_COUNTER.next()

  def __reduce__(self):
    return (DistArrayImpl, (self.shape, self.dtype, self.tiles, self.reducer_fn))

  def __del__(self):
    '''Destroy this array.

    NB: Destruction is actually deferred until the next usage of the
    blob_ctx.  __del__ can be called at anytime, including the
    invocation of a RPC call, which leads to odd/bad behavior.
    '''
    if self.ctx.worker_id == blob_ctx.MASTER_ID:
      #util.log_info('Destroying table... %s', self.id)
      ctx = self.ctx
      tiles = self.tiles.values()

      # drop reference to self from lambda
      self.ctx.defer(lambda: ctx.destroy_all(tiles))

  def id(self):
    return self.table.id()

  def extent_for_blob(self, id):
    return self.blob_to_ex[id]
  
  def tile_shape(self):
    scounts = collections.defaultdict(int)
    for ex in self.tiles.iterkeys():
      scounts[ex.shape] += 1
    
    return sorted(scounts.items(), key=lambda kv: kv[1])[-1][0]

  def map_to_table(self, mapper_fn, kw=None):
    ctx = blob_ctx.get()

    if kw is None: kw = {}
    kw['array'] = self
    kw['user_fn'] = mapper_fn

    return ctx.map(self.tiles.values(),
                   mapper_fn = _array_mapper,
                   reduce_fn=None,
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

    #util.log_info('FETCH: %s %s', self.shape, region)

    ctx = blob_ctx.get()
    assert np.all(region.lr <= self.shape), (region, self.shape)
    
    # special case exact match against a tile 
    if region in self.tiles:
      #util.log_info('Exact match.')
      ex, intersection = region, region
      blob_id = self.tiles[region]
      tgt = ctx.get(blob_id, extent.offset_slice(ex, intersection))
    else:
      splits = list(extent.find_overlapping(self.tiles.iterkeys(), region))

      #util.log_info('Target shape: %s, %d splits', region.shape, len(splits))
      tgt = np.ma.MaskedArray(np.ndarray(region.shape, dtype=self.dtype))
      tgt.mask = tile.MASK_INVALID
      #util.log_info('Fetching %d tiles', len(splits))

      futures = []
      for ex, intersection in splits:
        dst_slice = extent.offset_slice(region, intersection)
        blob_id = self.tiles[ex]

        def _apply_data(r, slc=dst_slice):
          tgt[slc] = r.data

        futures.append(ctx.get(blob_id, extent.offset_slice(ex, intersection),
                               callback=_apply_data))
        #tgt[dst_slice] = ctx.get(blob_id, extent.offset_slice(ex, intersection))

        #util.log_info('%s %s', dst_slice, src_slice.shape)
      rpc.wait_for_all(futures)

    # attempt to remove mask on arrays when it is
    # all valid.
    if isinstance(tgt, np.ma.MaskedArray) and np.all(tgt.mask == np.ma.nomask):
      return tgt.data

    return tgt
    #return tile.data[]
   
  def update_slice(self, slc, data):
    return self.update(extent.from_slice(slc, self.shape), data)
     
  def update(self, region, data):
    ctx = blob_ctx.get()
    Assert.isinstance(region, extent.TileExtent)
    Assert.eq(region.shape, data.shape,
              'Size of extent does not match size of data')

    #util.log_info('%s %s', self.table.id(), self.tiles)
    # exact match
    if region in self.tiles:
      #util.log_info('EXACT: %d %s ', self.table.id(), region)
      blob_id = self.tiles[region]
      ctx.update(blob_id, tile.from_data(data), self.reducer_fn)
      return
    
    splits = list(extent.find_overlapping(self.tiles, region))
    futures = []
    #util.log_info('Updating %s tiles', len(splits))
    for dst_key, intersection in splits:
      #util.log_info('%d %s %s %s', self.table.id(), region, dst_key, intersection)
      blob_id = self.tiles[dst_key]
      src_slice = extent.offset_slice(region, intersection)
      update_tile = tile.from_intersection(dst_key, intersection, data[src_slice])
      #util.log_info('%s %s %s %s', dst_key.shape, intersection.shape, blob_id, update_tile)
      futures.append(ctx.update(blob_id, update_tile, self.reducer_fn, wait=False))

    rpc.wait_for_all(futures)


def create(shape,
           dtype=np.float,
           sharder=None,
           combiner=None,
           reducer=None,
           tile_hint=None):
  '''Make a new, empty DistArray'''
  ctx = blob_ctx.get()
  dtype = np.dtype(dtype)
  shape = tuple(shape)

  extents = compute_splits(shape, tile_hint, -1).keys()
  tiles = {}
  for i, ex in enumerate(extents):
    tiles[ex] = ctx.create(tile.from_shape(ex.shape, dtype), hint=i)

  for ex in extents:
    tiles[ex] = tiles[ex].wait().blob_id

  return DistArrayImpl(shape=shape, dtype=dtype, tiles=tiles, reducer_fn=reducer)

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
    # fetch a one element array in order to get the dtype
    key, blob_id = extents.iteritems().next()
    #util.log_info('%s :: %s', key, blob_id)
    t = blob_ctx.get().get(blob_id, None)
    Assert.isinstance(t, tile.Tile)
    dtype = t.dtype
  else:
    # empty table; default dtype.
    dtype = np.float

  return DistArrayImpl(shape=shape, dtype=dtype, tiles=extents, reducer_fn=None)

class LocalWrapper(DistArray):
  '''
  Provide the `DistArray` interface for local data.
  '''
  def __init__(self, data):
    self._data = np.asarray(data)
    assert not isinstance(data, core.BlobId)
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
    if kw is None: kw = {}
    ex = extent.from_slice(np.index_exp[:], self.shape)
    result = mapper_fn(ex, self._data, **kw)
    assert len(result) == 1
    result_ex, result_data = result[0]

    return as_array(result_data)

  def __getitem__(self, idx):
    return self._data[idx]

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
    return []

  offset = extent.offset_from(slice_extent, intersection)
  offset.array_shape = slice_extent.shape

  subslice = extent.offset_slice(ex, intersection)

  result = mapper_fn(offset, **fn_kw)
  #util.log_info('Slice mapper[%s] %s %s -> %s', mapper_fn, offset, subtile, result)
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
    
  def map_to_table(self, mapper_fn, kw):
    return self.darray.map_to_table(mapper_fn = _slice_mapper,
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
  
  def __repr__(self):
    return 'Broadcast(%s -> %s)' % (self.base, self.shape)
  
  def fetch(self, ex):
    # drop extra dimensions
    while len(ex.shape) > len(self.base.shape):
      ex = extent.drop_axis(ex, -1)
      
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

    template = np.ndarray(ex.shape, dtype=self.base.dtype)
    fetched = self.base.fetch(ex)
    
    _, bcast = np.broadcast_arrays(template, fetched)
    return bcast 


def broadcast(args):
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
    axis_shape = set(s[axis] for s in new_shapes)
   
    assert len(axis_shape) <= 2, 'Mismatched shapes for broadcast: %s' % orig_shapes
    if len(axis_shape) == 2:
      assert 1 in axis_shape, 'Mismatched shapes for broadcast: %s' % orig_shapes
  
    # now lift the inputs with size(axis) == 1 
    # to have the maximum size for the axis 
    max_size = max(s[axis] for s in new_shapes)
    for s in new_shapes:
      s[axis] = max_size
    
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
    return np.prod(v.base.shape)
  return np.prod(v.shape)

def largest_value(vals):
  return sorted(vals, key=lambda v: _size(v))[-1]

