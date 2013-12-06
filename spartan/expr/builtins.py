'''
Basic numpy style operations on arrays.

These include --

* Array creation routines: (`rand`, `randn`, `zeros`, `ones`, `arange`)
* Reductions: (`sum`, `argmin`, `argmax`, `mean`)
* Shape/type casting: (`reshape`, `ravel`, `astype`, `shape`, `size`)
* Other: (`dot`).
'''
import numpy as np
from ..array import distarray, extent
from ..array.extent import index_for_reduction, shapes_match
from .base import force
from .shuffle import shuffle
from .map import map
from .ndarray import ndarray
from .reduce import reduce
from .loop import loop
from spartan import util
from ..util import Assert

def rand(*shape, **kw):
  '''
  :param tile_hint: A tuple indicating the desired tile shape for this array.
  '''
  for s in shape: assert isinstance(s, int)
  return map(ndarray(shape, 
                     dtype=np.float,
                     tile_hint=kw.get('tile_hint', None)),
                     fn = lambda input: np.random.rand(*input.shape))
  
def randn(*shape, **kw):
  for s in shape: assert isinstance(s, int)
  return map(ndarray(shape,
                     dtype=np.float,
                     tile_hint=kw.get('tile_hint', None)),
             fn = lambda input: np.random.randn(*input.shape))

def zeros(shape, dtype=np.float, tile_hint=None):
  return map(ndarray(shape, dtype=np.float, tile_hint=tile_hint),
             fn = lambda input: np.zeros(input.shape))

def ones(shape, dtype=np.float, tile_hint=None):
  return map(ndarray(shape, dtype=np.float, tile_hint=tile_hint), 
             fn = lambda input: np.ones(input.shape, dtype))

def _arange_mapper(inputs, ex, dtype=None):
  pos = extent.ravelled_pos(ex.ul, ex.array_shape)
  #util.log_info('Extent: %s, pos: %s', ex, pos)
  sz = np.prod(ex.shape)
  yield (ex, np.arange(pos, pos+sz, dtype=dtype).reshape(ex.shape))


def arange(shape, dtype=np.float):
  return shuffle(ndarray(shape, dtype=dtype), 
                     fn = _arange_mapper, 
                     kw = {'dtype' : dtype })

def _sum_local(ex, tile, axis):
  return np.sum(tile[:], axis)

def sum(x, axis=None):
  '''
  Sum ``x`` over ``axis``.
  
  
  :param x: The array to sum.
  :param axis: Either an integer or ``None``.
  '''
  return reduce(x, axis=axis,
                       dtype_fn = lambda input: input.dtype,
                       local_reduce_fn = _sum_local,
                       combine_fn = lambda a, b: a + b)
    
def mean(x, axis=None):
  '''
  Compute the mean of ``x`` over ``axis``.
  
  See `numpy.ndarray.mean`.
  
  :param x: `Expr`
  :param axis: integer or ``None``
  '''
  return sum(x, axis) / x.shape[axis]


def _to_structured_array(*vals):
  '''Create a structured array from the given input arrays.
  
  :param vals: A list of (field_name, `np.ndarray`)
  :rtype: A structured array with fields from ``kw``.
  '''
  out = np.ndarray(vals[0][1].shape, 
                  dtype=','.join([a.dtype.str for name, a in vals]))
  out.dtype.names = [name for name, a in vals]
  for k, v in vals:
    out[k] = v
  return out

def _take_idx_mapper(input):
  return input['idx']


def _dual_reducer(ex, tile, axis, idx_f=None, val_f=None):
  local_idx = idx_f(tile[:], axis)
  local_val = val_f(tile[:], axis)

  global_idx = ex.to_global(local_idx, axis)
  new_idx = index_for_reduction(ex, axis)
  new_val = _to_structured_array(('idx', global_idx), ('val', local_val))
  
  assert shapes_match(new_idx, new_val), (new_idx, new_val.shape)
  return new_val

def _dual_combiner(a, b, op):
  return np.where(op(a['val'], b['val']), a, b)

def _dual_dtype(input):
  dtype = np.dtype('i8,%s' % input.dtype.str)
  dtype.names = ('idx', 'val')
  return dtype 


def argmin(x, axis=None):
  '''
  Compute argmin over ``axis``.
  
  See `numpy.ndarray.argmin`.
  
  :param x: `Expr` to compute a minimum over. 
  :param axis: Axis (integer or None).
  '''
  compute_min = reduce(x, axis,
                       dtype_fn = _dual_dtype,
                       local_reduce_fn = _dual_reducer,
                       combine_fn = lambda a, b: _dual_combiner(a, b, np.less),
                       fn_kw={'idx_f' : np.argmin, 'val_f' : np.min})
  
  take_indices = map(compute_min, _take_idx_mapper)
  return take_indices

def argmax(x, axis=None):
  '''
  Compute argmax over ``axis``.
  
  See `numpy.ndarray.argmax`.
  
  :param x: `Expr` to compute a maximum over. 
  :param axis: Axis (integer or None).
  '''
  compute_max = reduce(x, axis,
                       dtype_fn = _dual_dtype,
                       local_reduce_fn = _dual_reducer,
                       combine_fn = lambda a, b: _dual_combiner(a, b, np.greater),
                       fn_kw={'idx_f' : np.argmax, 'val_f' : np.max})
  
  take_indices = map(compute_max, _take_idx_mapper)
  return take_indices
  

def size(x, axis=None):
  '''
  Return the size (product of the size of all axes) of ``x``.
  
  See `numpy.ndarray.size`.
  
  :param x: `Expr` to compute the size of.
  '''
  if axis is None:
    return np.prod(x.shape)
  return x.shape[axis]

def astype(x, dtype):
  '''
  Convert ``x`` to a new dtype.
  
  See `numpy.ndarray.astype`.
  
  :param x:
  :param dtype:
  '''
  assert x is not None
  return map(x, lambda tile: tile.astype(dtype))

def _ravel_mapper(array, ex):
  ul = extent.ravelled_pos(ex.ul, ex.array_shape)
  lr = 1 + extent.ravelled_pos([lr - 1 for lr in ex.lr], ex.array_shape)
  shape = (np.prod(ex.array_shape),)
  
  ravelled_ex = extent.create((ul,), (lr,), shape)
  ravelled_data = array.fetch(ex).ravel()
  yield ravelled_ex, ravelled_data
   
def ravel(v):
  '''
  "Ravel" ``v`` to a one-dimensional array of shape (size(v),).
  
  See `numpy.ndarray.ravel`.
  :param v:
  '''
  return shuffle(v, _ravel_mapper)

def _reshape_mapper(array, ex, _dest_shape):
  tile = array.fetch(ex)
  
  ravelled_ul = extent.ravelled_pos(ex.ul, ex.array_shape)
  ravelled_lr = extent.ravelled_pos([lr - 1 for lr in ex.lr], ex.array_shape)
  
  target_ul = extent.unravelled_pos(ravelled_ul, _dest_shape)
  target_lr = extent.unravelled_pos(ravelled_lr, _dest_shape)
  
  #util.log_info('%s + %s -> %s', ravelled_ul, _dest_shape, target_ul)
  #util.log_info('%s + %s -> %s', ravelled_lr, _dest_shape, target_lr)
  
  target_ex = extent.create(target_ul, np.array(target_lr) + 1, _dest_shape)
  yield target_ex, tile.reshape(target_ex.shape)

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
  
  return shuffle(array, 
                     _reshape_mapper, 
                     tile_hint=tile_hint,
                     kw = { '_dest_shape' : new_shape})

def _dot_mapper(inputs, ex, av, bv):
  # read current tile of array 'a'
  ex_a = ex

  # fetch corresponding column tile of array 'b'
  # rows = ex_a.cols
  # cols = *
  ex_b = extent.create((ex_a.ul[1], 0),
                       (ex_a.lr[1], bv.shape[1]),
                       bv.shape)

  util.log_info('%s %s', ex_a, ex_b)

  a = av.fetch(ex_a)
  b = bv.fetch(ex_b)

  result = np.dot(a, b)

  ul = np.asarray([ex_a.ul[0], 0])
  lr = ul + result.shape
  #util.log_info('%s %s %s', a.shape, b.shape, result.shape)
  #util.log_info('%s %s %s', ul, lr, target_shape)
  target_shape = (av.shape[0], bv.shape[1])
  out = extent.create(ul, lr, target_shape)
  
  yield out, result

def _dot_numpy(array, ex, numpy_data=None):
  l = array.fetch(ex)
  r = numpy_data
  
  yield (ex[0].add_dim(), np.dot(l, r))
  

def dot(a, b):
  '''
  Compute the dot product (matrix multiplication) of 2 arrays.
  
  :param a: `Expr` or `numpy.ndarray` 
  :param b: `Expr` or `numpy.ndarray`
  :rtype: `Expr`
  '''
  av = force(a)
  bv = force(b)
  
  if isinstance(bv, np.ndarray):
    return shuffle(av, _dot_numpy, kw = { 'numpy_data' : bv })
  
  #av, bv = distarray.broadcast([av, bv])
  Assert.eq(a.shape[1], b.shape[0])
  target = ndarray((a.shape[0], b.shape[1]),
                   dtype=av.dtype,
                   tile_hint=av.tile_shape(),
                   combine_fn=np.add,
                   reduce_fn=np.add)
  
  return shuffle(av, _dot_mapper, target=target, kw = dict(av=av, bv=bv))
            

def ln(v): return map(v, fn=np.log)
def log(v): return map(v, fn=np.log)
def exp(v): return map(v, fn=np.exp)
def sqrt(v): return map(v, fn=np.sqrt)
def abs(v): return map(v, fn=np.abs)

try:
  import scipy.stats
  def norm_cdf(v):
    return map(v, fn=scipy.stats.norm.cdf)
except:
  util.log_info('Missing scipy.stats (some functions will be unavailable.')
