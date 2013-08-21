#!/usr/bin/env python

from . import tile, extent
from spartan import pytable
from spartan.pytable import util
from spartan.util import Assert
import numpy as np


# number of elements per tile
TILE_SIZE = 100000

def find_shape(extents):
  util.log('Finding shape... %s', extents)
  return np.max([ex.lr for ex in extents], axis=0)


def get_data(data, index):
  if isinstance(data, tuple):
    return tuple([get_data(d, index) for d in data])
  if not isinstance(data, np.ndarray):
    data = np.array(data)
  if not data.shape:
    data = data.reshape((1,))
  return data[index]


def split_extent(array, tile_extent):
#   util.log('Splitting tile %s', chunk.extent)
  for ex in array.extents:
    intersection = extent.intersection(ex, tile_extent)
    if intersection is not None:
      yield (ex, intersection)
      

def split_tile(array, tile_extent, tile_data):
  for ex in split_extent(array, tile_extent):
    intersection = extent.intersection(ex, tile_extent)
    local_idx = tile_extent.local_offset(intersection)
    yield (ex, intersection, get_data(tile_data, local_idx))
       
  
def find_matching_tile(array, tile_extent):
  for ex in array.extents():
    ul_diff = tile_extent.ul - ex.ul
    lr_diff = ex.lr - tile_extent.lr
    if np.all(ul_diff >= 0) and np.all(lr_diff >= 0):
      # util.log('%s matches %s', ex, tile_extent)
      return array.tile_for_extent(ex)
  
  raise Exception, 'No matching tile_extent!' 

def extent_from_slice(array, slice):
  '''
  :param array: Distarray
  :param slice: A tuple of `slice` objects.
  :rtype: `Extent`
  '''
  ul = []
  sz = []
  for dim, slc in enumerate(slice):
    start, stop, step = slc.indices(array.shape[dim])
    ul.append(start)
    sz.append(stop - start)
  
  return extent.TileExtent(ul, sz, array.shape)


class TileAccum(object):
  def __init__(self, accum):
    self.accum = accum
  
  def __call__(self, old_tile, new_tile):
    assert isinstance(old_tile, tile.Tile), type(old_tile)
    assert isinstance(new_tile, tile.Tile), type(new_tile)
    
    if old_tile.data is None:
      old_tile._initialize()
    
    idx = old_tile.extent.local_offset(new_tile.extent)
    data = old_tile.data[idx]
    
    invalid = old_tile.mask[idx]
    valid = ~old_tile.mask[idx]
    data[invalid] = new_tile.data[invalid]
    old_tile.mask[invalid] = False
    if data[valid].size > 0:
      data[valid] = self.accum(data[valid], new_tile.data[valid])
      
def take_first(a,b):
  return a


accum_replace = TileAccum(take_first)
accum_min = TileAccum(np.minimum)
accum_max = TileAccum(np.maximum)
accum_sum = TileAccum(np.add)


  
class NestedSlice(object):
  def __init__(self, extent, subslice):
    self.extent = extent
    self.subslice = subslice
    
  def __hash__(self):
    return hash(self.extent)
  
  def __eq__(self, other):
    Assert.is_instance(other, extent.TileExtent)
    return self.extent == other 



class TileSelector(object):
  def __call__(self, k, v):
#     util.log('Selector called for %s', k)
    if isinstance(k, extent.TileExtent): 
      return v[:]
    if isinstance(k, NestedSlice):
      result = v[k.subslice]
#       print k.extent, k.subslice, result.shape
      return result
    raise Exception, "Can't handle type %s" % type(k)
  

def _compute_splits(shape):
  '''Split an array of shape ``shape`` into tiles containing roughly 
  `TILE_SIZE` elements.'''
  if len(shape) == 1:
    weight = 1
  else:
    weight, sub_splits = _compute_splits(shape[1:])
  
  my_splits = []
  step = max(1, TILE_SIZE / weight)
  for i in range(0, shape[0], step):
    my_dim = (i, min(shape[0], i + step))
    my_splits.append([my_dim])

  if len(shape) == 1:
    return (shape[0], my_splits)
  
  out = []
  
  for i in my_splits:
    for j in sub_splits:
      out.append(i + j)
  return (weight * shape[0], out)

def compute_splits(shape):
  return _compute_splits(shape)[1]


def _create_rand(extent, data):
  data[:] = np.random.rand(*extent.shape)

def _create_randn(extent, data):
  data[:] = np.random.randn(*extent.shape)
  
def _create_ones(extent, data):
#   util.log('Updating %s, %s', extent, data)
  data[:] = 1

def _create_zeros(extent, data):
  data[:] = 0

def _create_range(extent, data):
  pos = extent.ravelled_pos()
  sz = np.prod(extent.shape)
  data[:] = np.arange(pos, pos+sz).reshape(extent.shape)
  
class DistArray(object):
  def id(self):
    return self.table.id()
  
  @staticmethod
  def from_table(table):
    '''
    Construct a distarray from an existing table.
    Keys must be of type `Extent`, values of type `Tile`.
    
    Shape is computed as the maximum range of all extents.
    
    Dtype is taken from the dtype of the tiles.
    
    :param table:
    '''
    d = DistArray()
    d.table = table
    d.extents = {}
    
    keys = table.keys()
    util.log('KEYS: %s', keys)
    for extent in keys:
      assert not (extent in d.extents)
      d.extents[extent] = 1
    
    if not d.extents:
      d.shape = tuple()
    else:
      d.shape = find_shape(keys)
    
    
    if len(keys) > 0:
      tile = table[keys[0]]
      d.dtype = tile.dtype
    else:
      # empty table; default dtype.
      d.dtype = np.float
    
    return d
  
  @staticmethod
  def create(master, shape, dtype=np.float, 
             sharder=pytable.mod_sharder,
             accum=accum_replace):
    total_elems = np.prod(shape)
    splits = compute_splits(shape)
    util.log('Creating array with %d tiles', len(splits))
    extents = []
    for split in splits:
      ul, lr = zip(*split)
      sz = np.array(lr) - np.array(ul)
      extents.append(extent.TileExtent(ul, sz, shape))

    table = master.create_table(sharder, accum, TileSelector())
    for ex in extents:
      ex_tile = tile.make_tile(ex, None, dtype, masked=True)
      table.update(ex, ex_tile)
    
    d = DistArray()
    d.shape = shape
    d.table = table
    d.extents = extents
    return d
  
  @staticmethod
  def create_with(master, shape, init_fn):
    d = DistArray.create(master, shape)
    pytable.map_inplace(d.table, init_fn)
    return d 
  
  @staticmethod
  def randn(master, *shape):
    return DistArray.create_with(master, shape, _create_randn)
  
  @staticmethod
  def rand(master, *shape):
    return DistArray.create_with(master, shape, _create_rand)
  
  @staticmethod
  def ones(master, shape):
    return DistArray.create_with(master, shape, _create_ones)
  
  @staticmethod
  def arange(master, shape):
    return DistArray.create_with(master, shape, _create_range)
  
  def map(self, fn, *args):
    return DistArray.from_table(pytable.map_items(self.table, fn, *args))
  
  def map_tiles(self, fn, kw):
    return pytable.map_items(self.table, fn, kw)  
  
  def _get(self, extent):
    return self.table.get(extent)
  
  def ensure(self, region):
    '''
    Return a local numpy array for the given region.
    
    If necessary, data will be copied from remote hosts to fill the region.    
    :param region: `Extent` indicating the region to fetch.
    '''
    Assert.is_instance(region, extent.TileExtent)
    splits = list(split_extent(self, region))
    
    if len(splits) == 1:
      ex, intersection = splits[0]
      return self.table.get(NestedSlice(ex, ex.local_offset(intersection)))

    util.log('Target shape: %s', region.shape)
    tgt = np.ndarray(region.shape)
    for ex, intersection in splits:
      dst_slice = region.local_offset(intersection)
      src_slice = self.table.get(NestedSlice(ex, ex.local_offset(intersection)))
      #src_slice = self.table.get(ex)
      tgt[dst_slice] = src_slice
    return tgt
    #return tile.data[]
    
  def update(self, region, data):
    pass
  
  def __getitem__(self, idx):
    if isinstance(idx, int):
      return self[idx:idx + 1][0]
    if not isinstance(idx, tuple):
      idx = tuple(idx)
    if len(idx) < len(self.shape):
      idx = tuple(list(idx) + [slice(None, None, None) for _ in range(len(self.shape) - len(self.key))])
    
    ex = extent.TileExtent.from_slice(idx, self.shape)
    return self.ensure(ex)
  
  def glom(self):
    print self.shape
    ex = extent.TileExtent([0] * len(self.shape), self.shape, self.shape)
    return self.ensure(ex)

