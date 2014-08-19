import numpy as np
import scipy.sparse as sp
from ..array import extent
from .shuffle import shuffle

def _scan_reduce_mapper(array, ex, reduce_fn, axis):
  local_reduction = reduce_fn(array.fetch(ex), axis=axis)
  axis_shape = array.tile_shape()[axis]
  id = (ex.lr[axis]-1) / axis_shape
  new_ul = list(ex.ul)
  new_lr = list(ex.lr)
  new_shape = list(ex.array_shape)
  new_ul[axis] = id
  new_lr[axis] = id + 1
  new_shape[axis] = int(np.ceil(array.shape[axis] * 1.0 / axis_shape))

  dst_ex = extent.create(new_ul, new_lr, new_shape)

  local_reduction = np.asarray(local_reduction).reshape(dst_ex.shape)
  yield (dst_ex, local_reduction)

def _scan_mapper(array, ex, scan_fn, axis=None, scan_base=None):
  local_data = array.fetch(ex)
  if sp.issparse(local_data): local_data = local_data.todense()
  
  if axis is None:
    axis = 1
    id = (ex.lr[axis] - 1) / array.tile_shape()[axis]
    base_slice = list(ex.to_slice())
    base_slice[axis] = slice(id, id+1, None)
    new_slice = [slice(0, ex.shape[i], None) for i in range(len(ex.shape))]
    new_slice[axis] = slice(0,1,None)
    local_data[new_slice] += scan_base[base_slice]
  else:
    id = (ex.lr[axis] - 1) / array.tile_shape()[axis]
    if id > 0:
      base_slice = list(ex.to_slice())
      base_slice[axis] = slice(id-1, id, None)
      new_slice = [slice(0, ex.shape[i], None) for i in range(len(ex.shape))]
      new_slice[axis] = slice(0,1,None)
      local_data[new_slice] += scan_base[base_slice]
      
  yield (ex, np.asarray(scan_fn(local_data, axis=axis)).reshape(ex.shape))   
      
def scan(array, reduce_fn=np.sum, scan_fn=np.cumsum, axis=None):
  '''
  Scan ``array`` over ``axis``.
  
  
  :param array: The array to scan.
  :param reduce_fn: local reduce function
  :param scan_fn: scan function
  :param axis: Either an integer or ``None``.
  '''
  reduce_result = shuffle(array, fn=_scan_reduce_mapper, kw={'axis': axis if axis is not None else 1,
                                                             'reduce_fn': reduce_fn}, shape_hint=array.shape)
  fetch_result = reduce_result.optimized().glom()
  if axis is None:
    fetch_result = np.concatenate((np.zeros(1), scan_fn(fetch_result, axis=None)[:-1])).reshape(fetch_result.shape)
  else:
    fetch_result = scan_fn(fetch_result, axis=axis)
  scan_result = shuffle(array, fn=_scan_mapper, kw={'scan_fn':scan_fn,
                                                    'axis': axis,
                                                    'scan_base':fetch_result})
  return scan_result