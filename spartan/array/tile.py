import numpy as np
import numpy.ma as ma
import scipy.sparse as sp
from _ctile_py_if import TileBase
from spartan import util
from spartan.array import sparse

TYPE_DENSE = TileBase.TILE_DENSE
TYPE_MASKED = TileBase.TILE_MASKED
TYPE_SPARSE = TileBase.TILE_SPARSE

builtin_reducers = {}
builtin_reducers[None] = TileBase.TILE_REDUCER_REPLACE
builtin_reducers[np.add] = TileBase.TILE_REDUCER_ADD
builtin_reducers[np.multiply] = TileBase.TILE_REDUCER_MUL
builtin_reducers[np.maximum] = TileBase.TILE_REDUCER_MAXIMUM
builtin_reducers[np.minimum] = TileBase.TILE_REDUCER_MINIMUM


def npdata_to_internal(array):
  ttype = None
  stype = TileBase.TILE_SPARSE_NONE
  data = None
  if sp.issparse(array):
    ttype = TileBase.TILE_SPARSE
    if isinstance(array, sp.coo_matrix):
      stype = TileBase.TILE_SPARSE_COO
      data = (array.row, array.col, array.data)
      util.log_info('npdate_to_internal %s %s %s',
                    str(array.row.shape),
                    str(array.col.shape),
                    str(array.data.shape))
    elif isinstance(array, sp.csc_matrix):
      stype = TileBase.TILE_SPARSE_CSC
      data = (array.indices, array.indptr, array.data)
      util.log_info('npdate_to_internal %s %s %s',
                    str(array.indices.shape),
                    str(array.indptr.shape),
                    str(array.data.shape))
    elif isinstance(array, sp.csr_matrix):
      stype = TileBase.TILE_SPARSE_CSR
      data = (array.indices, array.indptr, array.data)
      util.log_info('npdate_to_internal %s %s %s',
                    str(array.indices.shape),
                    str(array.indptr.shape),
                    str(array.data.shape))
    else:
      stype = TileBase.TILE_SPARSE_COO
      array = array.tocoo()
      data = (array.row, array.col, array.data)
  elif isinstance(array, ma.MaskedArray):
    ttype = TileBase.TILE_MASKED
    data = (array.data, array.mask)
  else:
    ttype = TileBase.TILE_DENSE
    data = (array,)

  return array.shape, array.dtype.char, ttype, stype, data


class Tile(TileBase):
  ''' Wrapper class of Tile.
      If a operation needs to communicate with Python, NumPy and Scipy,
      it is pointless to implemented in pure c++. This class provides
      simpler implementation for such operations.
  '''
  def __init__(self, shape, dtype, tile_type, sparse_type, data, ctile_id=None):
    util.log_info('Tile.__init__')
    super(Tile, self).__init__(shape, np.dtype(dtype).char, tile_type,
                               sparse_type, data, ctile_id=ctile_id)

  def get(self, subslice, local=False):
    if local:
      return self.data[subslice]
    else:
      return super(Tile, self).get(subslice)

  def update(self, subslice, data, reducer):
    _, _, tile_type, sparse_type, tile_data = npdata_to_internal(data)

    if builtin_reducers.get(reducer, None) is None:
      _reducer = reducer
    else:
      _reducer = builtin_reducers[reducer]
    self._update(subslice, tile_type, sparse_type,
                 tile_data, _reducer)
    return self

  @property
  def dtype(self):
    return np.dtype(super(Tile, self).dtype)


def from_data(data):
  util.log_info("from_data %s", str(type(data)))
  if not isinstance(data, np.ndarray) and not sp.issparse(data):
    data = np.asarray(data)
  shape, dtype, tile_type, sparse_type, tile_data = npdata_to_internal(data)

  assert isinstance(tile_data, tuple), (type(tile_data))
  return Tile(shape,
              dtype,
              tile_type,
              sparse_type,
              tile_data)


def from_shape(shape, dtype, tile_type, sparse_type=TileBase.TILE_SPARSE_COO):
  util.log_info("from_shape")
  return Tile(shape,
              np.dtype(dtype).char,
              tile_type,
              sparse_type,
              None)


# DONT use this API. This API is only for internal usage.
def _internal_update(old_tile, subslice, update, reducer):
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
    if isinstance(old_tile) == np.ndarray:
      assert False, "dense = reducer(dense[subslice], sparse) doesn't support customized reducer."
    else:
      if old_tile.shape == update.shape:
        old_tile = reducer(old_tile.data, update)
      else:
        old_tile = sparse.compute_sparse_update(old_tile.data, update, subslice, reducer)
  # Apply a dense update array to the current tile data (which may be sparse or dense)
  else:
    if isinstance(old_tile) == np.ndarray:
      if old_tile.shape == update.shape:
        old_tile = reducer(old_tile, update)
      else:
        old_tile[subslice] = reducer(old_tile[subslice], update)
    else:
      # TODO (SPARSE UPDATE)!!!
      # sparse update, no mask, just iterate over data items
      # TODO(POWER) -- this is SLOW!
      assert False
      #if old_tile is not None and old_tile.data.format == 'coo':
        #old_tile = old_tile.data.tolil()
      #assert sp.issparse(update)
      #old_tile[subslice] = reducer(old_tile[subslice], update)

  return old_tile
