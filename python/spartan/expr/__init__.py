from ..dense import extent
from ..dense.extent import index_for_reduction, shapes_match
from ..util import Assert
from .map_extents import map_extents
from .map_tiles import map_tiles
from .ndarray import ndarray
from .outer import outer
from .reduce_extents import reduce_extents
from base import Expr, evaluate, dag, glom, eager, lazify, force
from spartan import util
import numpy as np

def map(v, fn, axis=None, **kw):
  return map_tiles(v, fn, **kw)

def rand(*shape, **kw):
  '''
  :param tile_hint: A tuple indicating the desired tile shape for this array.
  '''
  return map_tiles(ndarray(shape, 
                           dtype=np.float, 
                           tile_hint=kw.get('tile_hint', None)), 
                     fn = lambda inputs: np.random.rand(*inputs[0].shape))
  
def randn(*shape, **kw):
  return map_tiles(ndarray(shape, 
                           dtype=np.float, 
                           tile_hint=kw.get('tile_hint', None)), 
                     fn = lambda inputs: np.random.randn(*inputs[0].shape))

def zeros(shape, dtype=np.float, tile_hint=None):
  return map_tiles(ndarray(shape, dtype=np.float, tile_hint=tile_hint), 
                     fn = lambda inputs: np.zeros(inputs[0].shape))

def ones(shape, dtype=np.float, tile_hint=None):
  return map_tiles(ndarray(shape, dtype=np.float, tile_hint=tile_hint), 
                   fn = lambda inputs: np.ones(inputs[0].shape, dtype))

def _sum_local(index, tile, axis):
  return np.sum(tile[:], axis)

def _sum_reducer(a, b):
  return a + b

def sum(x, axis=None):
  return reduce_extents(x, axis=axis,
                       dtype_fn = lambda input: input.dtype,
                       local_reduce_fn = _sum_local,
                       combine_fn = lambda a, b: a + b)
    

def _to_structured_array(**kw):
  '''Create a structured array from the given input arrays.'''
  out = np.ndarray(kw.values()[0].shape, 
                  dtype=','.join([a.dtype.str for a in kw.itervalues()]))
  out.dtype.names = kw.keys()
  for k, v in kw.iteritems():
    out[k] = v
  return out


def _argmin_local(index, tile, axis):
  local_idx = np.argmin(tile[:], axis)
  local_min = np.min(tile[:], axis)

#  util.log_info('Index for reduction: %s %s %s',
#           index.array_shape,
#           axis,
#           index_for_reduction(index, axis))

  global_idx = index.to_global(local_idx, axis)

  new_idx = index_for_reduction(index, axis)
  new_value = _to_structured_array(idx=global_idx, min=local_min)

#   print index, value.shape, axis
#   print local_idx.shape
  assert shapes_match(new_idx, new_value), (new_idx, new_value.shape)
  return new_value

def _argmin_reducer(a, b):
  reduced = np.where(a['min'] < b['min'], a, b)
  return reduced

def _take_idx_mapper(inputs):
  return inputs[0]['idx']
 
def _argmin_dtype(input):
  dtype = np.dtype('i8,%s' % input.dtype.str)
  dtype.names = ('idx', 'min')
  return dtype 

def argmin(x, axis=None):
  x = x.force()
  compute_min = reduce_extents(x, axis,
                               dtype_fn = _argmin_dtype,
                               local_reduce_fn = _argmin_local,
                               combine_fn = _argmin_reducer)
  
  take_indices = map_tiles(compute_min, _take_idx_mapper)
  return take_indices
  

def size(x):
  return np.prod(x.shape)

def mean(x, axis=None):
  return sum(x, axis) / x.shape[axis]

def astype(x, dtype):
  assert x is not None
  return map_tiles(x, lambda inputs: inputs[0].astype(dtype))

def _ravel_mapper(inputs, ex):
  assert len(inputs) == 1
  ul = extent.ravelled_pos(ex.ul, ex.array_shape)
  lr = 1 + extent.ravelled_pos(ex.lr_array - 1, ex.array_shape)
  shape = (np.prod(ex.array_shape),)
  
  ravelled_ex = extent.create((ul,), (lr,), shape)
  ravelled_data = inputs[0].fetch(ex).ravel()
  return ravelled_ex, ravelled_data
   
def ravel(v):
  return map_extents((v,), _ravel_mapper)

def _reshape_mapper(inputs, ex, _dest_shape):
  tile = inputs[0].fetch(ex)
  
  ravelled_ul = extent.ravelled_pos(ex.ul, ex.array_shape)
  ravelled_lr = extent.ravelled_pos(ex.lr_array - 1, ex.array_shape)
  
  target_ul = extent.unravelled_pos(ravelled_ul, _dest_shape)
  target_lr = extent.unravelled_pos(ravelled_lr, _dest_shape)
  
  util.log_info('%s + %s -> %s', ravelled_ul, _dest_shape, target_ul)
  util.log_info('%s + %s -> %s', ravelled_lr, _dest_shape, target_lr)
  
  target_ex = extent.create(target_ul, np.array(target_lr) + 1, _dest_shape)
  return target_ex, tile.reshape(target_ex.shape)

def reshape(array, new_shape, tile_hint=None):
  '''
  Reshape/retile ``array``.
  
  Returns a new array with the given shape and tile size.
  :param array: `Expr`
  :param new_shape: `tuple`
  :param tile_hint: `tuple` or None
  '''
  
  old_size = np.prod(array.shape)
  new_size = np.prod(new_shape)
  
  Assert.eq(old_size, new_size, 'Size mismatch')
  
  return map_extents((array,), 
                     _reshape_mapper, 
                     tile_hint=tile_hint,
                     _dest_shape=new_shape) 

Expr.outer = outer
Expr.sum = sum
Expr.mean = mean
Expr.astype = astype
Expr.ravel = ravel
Expr.argmin = argmin


def _dot_mapper(inputs, ex):
  ex_a = ex
  # read current tile of array 'a'
  a = inputs[0].fetch(ex_a)

  target_shape = (inputs[0].shape[0], inputs[1].shape[1])
  
  # fetch corresponding column tile of array 'b'
  # rows = ex_a.cols
  # cols = *
  ex_b = extent.create((ex_a.ul[1], 0),
                       (ex_a.lr[1], inputs[1].shape[1]),
                       inputs[1].shape)
  b = inputs[1].fetch(ex_b)
  result = np.dot(a, b)
  
  ul = np.asarray([ex_a.ul[0], 0])
  lr = ul + result.shape
  util.log_info('%s %s %s', a.shape, b.shape, result.shape)
  util.log_info('%s %s %s', ul, lr, target_shape)
  out = extent.create(ul, lr, target_shape)
  
  return out, result

def _dot_numpy(inputs, ex, numpy_data=None):
  l = inputs[0].fetch(ex)
  r = numpy_data
  
  return (ex[0].add_dim(), np.dot(l, r))
  

def dot(a, b):
  av = force(a)
  bv = force(b)
  
  if isinstance(bv, np.ndarray):
    return map_extents((av,), _dot_numpy, numpy_data=bv)
  
  #av, bv = distarray.broadcast([av, bv])
  Assert.eq(a.shape[1], b.shape[0])
  target = ndarray((a.shape[0], b.shape[1]),
                   dtype=av.dtype,
                   tile_hint=av.tile_shape(),
                   combine_fn=np.add,
                   reduce_fn=np.add)
  
  return map_extents((av, bv), 
                     _dot_mapper,
                     target=target)
            

def _arange_mapper(inputs, ex, dtype=None):
  pos = extent.ravelled_pos(ex.ul, ex.array_shape)
  #util.log_info('Extent: %s, pos: %s', ex, pos)
  sz = np.prod(ex.shape)
  return (ex, np.arange(pos, pos+sz, dtype=dtype).reshape(ex.shape))


def arange(shape, dtype=np.float):
  return map_extents(ndarray(shape, dtype=dtype), 
                     fn = _arange_mapper,
                     dtype=dtype)
