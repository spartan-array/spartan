import traceback
import numpy as np
import scipy.sparse
import itertools
from . import extent
from spartan import util
from spartan.util import Assert
from spartan import sparse

TYPE_EMPTY = 0
TYPE_DENSE = 1
TYPE_MASKED = 2
TYPE_SPARSE = 3

MASK_ALL_CLEAR = 0
MASK_ALL_SET = 1

# get: slice -> ndarray or sparse or masked
# update: right now -- takes a Tile
#   change to update: takes a (slice, data, reducer) 
#   where data is dense, masked, or sparse.

ID = iter(xrange(100000000))

class Tile(object):
  '''
  Tiles have 4 modes:

  Empty   -- no data or mask
  Masked  -- data + mask
  Sparse  -- hashmap of positions (implicit mask)
  Dense   -- all data values have been set, mask is cleared.
  '''

  def __init__(self, shape, dtype, data, mask, tile_type):
    #Assert.ne(dtype, object)
    self.id = ID.next()
    self.shape = shape
    self.dtype = dtype
    self.type = tile_type
    self.mask = mask
    self.data = data

    if data is not None:
      Assert.eq(data.shape, shape)
      Assert.eq(data.dtype, dtype)

    if isinstance(mask, np.ndarray):
      Assert.eq(mask.shape, shape)

  @property
  def data(self):
    #util.log_info('DATA %s %s', self.id, self._data)
    return self._data

  @data.setter
  def data(self, val):
    #util.log_info('SET DATA %s %s', self.id, val)
    if val is not None:
      Assert.eq(val.dtype, self.dtype)

    self._data = val
    
  def update(self, subslice, data, reducer):
    #util.log_info('Update: %s %s', subslice, data)
    return merge(self, subslice, data, reducer)

  def get(self, subslice=None):
    # no data, return an empty array
    if self.data is None:
      #self._initialize()
      #util.log_info('EMPTY %s %s', self.id, self.shape)
      if self.type == TYPE_SPARSE:
        shape = self.shape if subslice == None else tuple([slice.stop - slice.start for slice in subslice])
        return scipy.sparse.coo_matrix(shape, self.dtype)
        
      return np.ndarray(self.shape, self.dtype)[subslice]

    # scalars are just returned directly.
    if len(self.data.shape) == 0:
      return self.data

    Assert.le(len(subslice), len(self.data.shape),
              'Selector has more dimensions than data! %s %s' % (subslice, self.data.shape))

    # otherwise if sparse, return a sparse subset
    if self.type == TYPE_SPARSE:
      if subslice is None or extent.is_complete(self.data.shape, subslice):
        result = self.data
      else:
        # TODO -- slicing doesn't really work for sparse arrays!
        util.log_warn('Trying to slice a sparse tile  -- this will likely fail!')
        result = self.data[subslice]
      return result
    
    # dense, check our mask and return a masked segment or unmasked if
    # the mask is all filled for the selected region.
    self._initialize_mask()
    
    #return self.data[subslice]
    if self.mask is None or np.all(self.mask[subslice]):
      #util.log_info('%s %s %s', self.data, self.mask, subslice)
      return self.data[subslice]
      
    data = self.data[subslice]
    mask = self.mask[subslice]
    result = np.ma.masked_all(data.shape, dtype=data.dtype)
    result[mask] = data[mask]

    #util.log_info('%s %s', self.id, result)
    return result

  def _initialize(self):
    # if this is a scalar, then data is just the scalar value itself.
    if len(self.shape) == 0:
      return

    if self.type == TYPE_SPARSE:
      if self.data is None:
        #util.log_info('New sparse: %s', self.shape)
        self.data = scipy.sparse.coo_matrix(self.shape, dtype=self.dtype)
    else:
      if self.data is None:
        self.data = np.zeros(self.shape, dtype=self.dtype)
      self._initialize_mask()

  def _initialize_mask(self):
    if self.type == TYPE_SPARSE:
      self.mask = None
      return
    
    if not isinstance(self.mask, np.ndarray):
      if self.mask == MASK_ALL_SET:
        self.mask = np.ones(self.shape, dtype=np.bool)
      elif self.mask == MASK_ALL_CLEAR:
        self.mask = np.zeros(self.shape, dtype=np.bool)


  def __repr__(self):
    return 'tile(%s, %s) [%s, %s]' % (self.shape, self.dtype, type(self.data), self.mask)


def from_data(data):
  if scipy.sparse.issparse(data):
    return Tile(
      shape=data.shape,
      data=data,
      dtype=data.dtype,
      mask=MASK_ALL_SET,
      tile_type=TYPE_SPARSE)
  else:
    return Tile(
      shape=data.shape,
      data=data,
      dtype=data.dtype,
      mask=np.ones(data.shape, dtype=np.bool),
      tile_type=TYPE_DENSE)


def from_shape(shape, dtype, tile_type):
  if tile_type == TYPE_SPARSE:
    return Tile(shape=shape,
                data=None,
                dtype=dtype,
                tile_type=TYPE_SPARSE,
                mask=None)
  elif tile_type == TYPE_DENSE: 
    return Tile(shape=shape,
                data=None,
                dtype=dtype,
                tile_type=tile_type,
                mask=MASK_ALL_CLEAR)
  else:
    assert False, 'Unknown tile type %s' % tile_type


def from_intersection(src, overlap, data):
  '''
  Return a tile for ``src``, masked to update the area specifed by ``overlap``.
  
  :param src: `TileExtent`
  :param overlap: `TileExtent`
  :param data:
  '''
  util.log_debug('%s %s %s', src, overlap, data.dtype)
  slc = extent.offset_slice(src, overlap)
  tdata = np.ndarray(src.shape, data.dtype)
  tmask = np.zeros(src.shape, np.bool)
  tmask[slc] = 1
  tdata[slc] = data
  return Tile(dtype=data.dtype,
              data=tdata,
              shape=src.shape,
              mask=tmask,
              tile_type=TYPE_DENSE)


def merge(old_tile, subslice, update, reducer):
  Assert.isinstance(old_tile, Tile)

  # TODO(Qi) -- see if we can still do the fast path.
  if old_tile.data is None:
    # return tile.from_data_and_shape(???)
    old_tile._initialize()

  #util.log_info('OLD: %s %s', old_tile.id, old_tile.data)
  #util.log_info('NEW: %s %s', new_tile.id, new_tile.data)

  # zero-dimensional arrays; just use data == None as a mask.
  if len(old_tile.shape) == 0:
    if old_tile.data is None or reducer is None:
      old_tile.data = update
    else:
      old_tile.data = reducer(old_tile.data, update)
    return old_tile

  assert not isinstance(update, np.ma.MaskedArray)
 
  # Apply a sparse update array to the current tile data (which may be sparse or dense)
  if scipy.sparse.issparse(update):
    if old_tile.type == TYPE_DENSE:
      #util.log_debug('Update sparse to dense')
      update_coo = update.tocoo()
      sparse.sparse_to_dense_update(old_tile.data, old_tile.mask, update_coo.row, update_coo.col, update_coo.data,
                                        sparse.REDUCE_ADD)
      #util.log_info('Update %s', update)
      #util.log_info('Update COO %s', update_coo) 
      #util.log_info('New mask: %s', old_tile.mask)
      #if reducer is not None:
      #  old_tile.data[subslice] = reducer(old_tile.data[subslice], update)
      #else:
      #  old_tile.data[subslice] = update.todense()
      old_tile.mask[subslice] = True
    else:
      if old_tile.shape == update.shape:
        if reducer is not None:
          old_tile.data = reducer(old_tile.data, update)
        else:
          old_tile.data = update
      else:
        old_tile.data = sparse.compute_sparse_update(old_tile.data,
                                                     update,
                                                     subslice,
                                                     reducer)
    return old_tile
    
  # Apply a dense update array to the current tile data (which may be sparse or dense)
  if old_tile.type == TYPE_DENSE:
    # initialize:
    # old_data[subslice] = data
    #
    # accumulate:
    # old_data[subslice] = reduce(old_data[subslice], data) 

    #util.log_info('%s %s', old_tile.mask, new_tile.mask)
    #util.log_info('REPLACE: %s', replaced)
    #util.log_info('UPDATE: %s', updated)
    
    # If the update shape is the same as the tile, 
    # then avoid doing a (possibly expensive) slice update. 
    if old_tile.data.shape == update.shape:
      if reducer is not None:
        old_tile.data = reducer(old_tile.data, update)
      else:
        old_tile.data = update
      old_tile.mask = np.ones(old_tile.shape, dtype=np.bool)
    else:
      replaced = ~old_tile.mask[subslice]
      updated = old_tile.mask[subslice]
    
      old_region = old_tile.data[subslice]  
      if np.any(replaced):  
        old_region[replaced] = update[replaced]
    
      if np.any(updated):
        if reducer is not None:
          old_region[updated] = reducer(old_region[updated], update[updated]) 
        else:
          old_region[updated] = update[updated] 
    
      old_tile.mask[subslice] = True
  else:
    if old_tile.data is not None: #and old_tile.data.format == 'coo':
      old_tile.data = old_tile.data.tolil() 
    #util.log_info('Update dense to sparse')
    # TODO (SPARSE UPDATE)!!!
    # sparse update, no mask, just iterate over data items
    # TODO(POWER) -- this is SLOW!
    if reducer is not None:
      old_tile.data[subslice] = reducer(old_tile.data[subslice], update)
    else:
      old_tile.data[subslice] = update

  return old_tile
