#!/usr/bin/env python

from . import extent, tile
from .. import util
from ..util import Assert
import numpy as np
import pytable


# number of elements to per dimension of a tile
TILE_SIZE = 100

def find_shape(extents):
  return np.max([ex.lr for ex in extents])


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
  for ex in array.extents():
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


class TileSelector(object):
  def __call__(self, k, v):
    if isinstance(k, extent.TileExtent): return v
    raise Exception


def compute_splits(shape):
  my_splits = []
  for i in range(0, shape[0], TILE_SIZE):
    my_dim = (i, min(shape[0], i + TILE_SIZE))
    my_splits.append([my_dim])

  if len(shape) == 1:
    return my_splits
  
  sub_splits = compute_splits(shape[1:])
  out = []
  
  for i in my_splits:
    for j in sub_splits:
      out.append(i + j)
  return out

def create_rand(extent, data):
  data[:] = np.random.randn(*extent.shape)
  return []
  
def create_ones(extent, data):
  data[:] = np.random.randn(*extent.shape)
  return []

  
class DistArray(object):
  def id(self):
    return self.table.id()
  
  
  @staticmethod
  def from_table(table):
    d = DistArray()
    d.table = table
    d.extents = {}
    
    keys = pytable.keys(table)
    for extent, _ in keys:
      assert not (extent in d.extents)
      d.extents[extent] = 1
    
    if not d.extents:
      d.shape = tuple()
    else:
      d.shape = find_shape(d.extents.keys())
    
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
    return DistArray.create_with(master, shape, create_rand)
  
  @staticmethod
  def ones(master, *shape):
    return DistArray.create_with(master, shape, create_ones)
  
  def map(self, fn, *args):
    return DistArray.from_table(pytable.map_items(self.table, fn, *args))
  
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
      tile = self.get(ex)
      return tile.data[ex.local_offset(intersection)]
    
    raise Exception
  
  def __getitem__(self, key):
    if isinstance(key, int):
      return self[key:key + 1][0]
    if not isinstance(key, tuple):
      key = tuple(key)
    if len(key) < len(self.shape):
      key = tuple(list(key) + [slice(None, None, None) for _ in range(len(self.shape) - len(self.key))])
    
    ex = extent.TileExtent.from_slice(self, key)
    return self.ensure(ex)
