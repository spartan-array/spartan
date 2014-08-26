import numpy as np
import numpy.ma as ma
import scipy.sparse as sp
from _ctile_py_if import *

TYPE_DENSE = Tile.TILE_DENSE
TYPE_MASKED = Tile.TILE_MASKED
TYPE_SPARSE = Tile.TILE_SPARSE

def _npdata_to_tuple (data):
  ret = []
  if sp.issparse(data):
    ret.append(Tile.TILE_SPARSE)
    if isinstance(data, sp.coo_matrix):
      ret.append((data.row, data.col, data.data))
      ret.append(Tile.TILE_SPARSE_COO)
    elif isinstance(data, sp.csc_matrix):
      ret.append((data.indices, data.inptr, data.data))
      ret.append(Tile.TILE_SPARSE_CSC)
    elif isinstance(data, sp.csr_matrix):
      ret.append((data.indices, data.inptr, data.data))
      ret.append(Tile.TILE_SPARSE_CSR)
  elif isinstance(data, ma.MaskedArray):
    ret.append(Tile.TILE_MASKED)
    ret.append((data.data, data.mask))
  else:
    ret.append(Tile.TILE_DENSE)
    ret.append((data,))

  return ret

class Tile(TileBase):
  ''' Wrapper class of Tile.
      If a operation needs to communicate with Python, NumPy and Scipy,
      it is pointless to implemented in pure c++. This class provides
      simpler implementation for such operations.
  '''
  def __init__(self, shape, dtype, tile_type, sparse_type, data):
    print '__init__', dtype
    super(Tile, self).__init__(shape, np.dtype(dtype).char, tile_type, sparse_type, data)
    self.builtin_reducers = {}
    self.builtin_reducers[np.add] = Tile.TILE_REDUCER_ADD
    self.builtin_reducers[np.multiply] = Tile.TILE_REDUCER_MUL
    self.builtin_reducers[np.maximum] = Tile.TILE_REDUCER_MAXIMUM
    self.builtin_reducers[np.minimum] = Tile.TILE_REDUCER_MINIMUM

  def get(self, subslice, local=False):
    if local:
      return self.data[subslice]
    else:
      return super(Tile, self).get(subslice)

  def update(self, subslice, data, reducer):
    print 'Tile.update'
    internal_data = _npdata_to_tuple(data)
    tile_type = internal_data[0]
    if len(internal_data) == 2:
      sparse_type = Tile.TILE_SPARSE_NONE
      tile_data = internal_data[1]
    else:
      sparse_type = internal_data[1]
      tile_data = internal_data[2]

    if self.builtin_reducers.get(reducer, None) is None:
      _reducer = reducer
    else:
      _reducer = self.builtin_reducers[reducer]
    print "DATATATATATATATATATATA", data, self
    self._update(subslice, tile_type, sparse_type,
                 tile_data, _reducer)
    return self

  @property
  def dtype(self):
    return np.dtype(super(Tile, self).dtype)

def from_data(data):
  internal_data = _npdata_to_tuple(data)
  tile_type = internal_data[0]
  tile_data = internal_data[1]

  if len(internal_data) < 3:
    sparse_type = Tile.TILE_SPARSE_NONE
  else:
    sparse_type = internal_data[2]

  shape = data.shape
  print 'from_data', data.dtype
  dtype = data.dtype.char

  assert isinstance(tile_data, tuple), (type(tile_data))
  return Tile(shape,
              dtype,
              tile_type,
              sparse_type,
              tile_data)


def from_shape(shape, dtype, tile_type, sparse_type=Tile.TILE_SPARSE_COO):
  print 'well', shape, dtype, tile_type, sparse_type
  return Tile(shape,
              np.dtype(dtype).char,
              tile_type,
              sparse_type,
              None)

# DONT use this API. This API is only for internal usage.
def _internal_update(data, subslice, update, reducer):
  # zero-dimensional arrays; just use data == None as a mask.
  if len(old_tile.shape) == 0:
    if old_tile.data is None or reducer is None:
      old_tile.data = update
    else:
      old_tile.data = reducer(old_tile.data, update)
    return old_tile

  assert not isinstance(update, ma.MaskedArray)

  # Apply a sparse update array to the current tile data (which may be sparse or dense)
  if sp.issparse(update):
    if isinstance(data) == np.ndarray:
      assert False, "dense = reducer(dense[subslice], sparse) doesn't support customized reducer."
    else:
      if data.shape == update.shape:
        data = reducer(old_tile.data, update)
      else:
        data = compute_sparse_update(old_tile.data, update, subslice, reducer)
  # Apply a dense update array to the current tile data (which may be sparse or dense)
  else:
    if isinstance(data) == np.ndarray:
      if data.shape == update.shape:
        data = reducer(data, update)
      else:
        data[subslice] = reducer(data[subslice], update)
    else:
      # TODO (SPARSE UPDATE)!!!
      # sparse update, no mask, just iterate over data items
      # TODO(POWER) -- this is SLOW!
      if data is not None: #and old_tile.data.format == 'coo':
        data = old_tile.data.tolil()
      assert sp.issparse(update)
      data[subslice] = reducer(data[subslice], update)

  return data
