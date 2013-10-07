'''Lazy arrays.

Expr operations are not performed immediately, but are set aside
and built into a control flow graph, which is then compiled
into a series of primitive array operations.
'''

from .node import Node
from prims import NotShapeable
from spartan import util
from spartan.dense import extent, distarray
from spartan.dense.extent import index_for_reduction, shapes_match
from spartan.util import Assert
import numpy as np
import spartan
import types

def _apply_binary_op(inputs, binary_op=None):
  assert len(inputs) == 2
  return binary_op(*inputs)

class Expr(object):
  _dag = None
  
  def __add__(self, other):
    return map_tiles((self, other), _apply_binary_op, binary_op=np.add)

  def __sub__(self, other):
    return map_tiles((self, other), _apply_binary_op, binary_op=np.subtract)

  def __mul__(self, other):
    return map_tiles((self, other), _apply_binary_op, binary_op=np.multiply)

  def __mod__(self, other):
    return map_tiles((self, other), _apply_binary_op, binary_op=np.mod)

  def __div__(self, other):
    return map_tiles((self, other), _apply_binary_op, binary_op=np.divide)

  def __eq__(self, other):
    return map_tiles((self, other), _apply_binary_op, binary_op=np.equal)

  def __ne__(self, other):
    return map_tiles((self, other), _apply_binary_op, binary_op=np.not_equal)

  def __lt__(self, other):
    return map_tiles((self, other), _apply_binary_op, binary_op=np.less)

  def __gt__(self, other):
    return map_tiles((self, other), _apply_binary_op, binary_op=np.greater)

  def __pow__(self, other):
    return map_tiles((self, other), _apply_binary_op, binary_op=np.power)

  def __getitem__(self, idx):
    return IndexExpr(src=self, idx=lazify(idx))

  def __setitem__(self, k, val):
    raise Exception, '__setitem__ not supported.'
  
  @property
  def shape(self):
    if hasattr(self, '_shape'):
      return self._shape
    
    try:
      dag = self.dag()
      return dag.shape()
    except NotShapeable:
      return evaluate(self).shape
    
  def dag(self):
    return dag(self)
  
  def evaluate(self):
    return evaluate(self) 

  def glom(self):
    return glom(self)


Expr.__rsub__ = Expr.__sub__
Expr.__radd__ = Expr.__add__
Expr.__rmul__ = Expr.__mul__
Expr.__rdiv__ = Expr.__div__


class LazyVal(Expr, Node):
  _members = ['val']
  
  def __reduce__(self):
    return self.evaluate().__reduce__()


def lazify(val):
  if isinstance(val, Expr): return val
  #util.log('Lazifying... %s', val)
  return LazyVal(val)


def val(x):
  return lazify(x)


def glom(node):    
  '''
  Evaluate this expression and return the result as a `numpy.ndarray`. 
  '''
  if isinstance(node, Expr):
    node = evaluate(node)
  
  if isinstance(node, np.ndarray):
    return node
  
  return node.glom()


def dag(node):
  if not isinstance(node, Expr):
    raise TypeError
    
  if node._dag is not None:
    return node._dag
  
  from . import compile_expr
  dag = compile_expr.compile(node)
  dag = compile_expr.optimize(dag)
  node._dag = dag
  return node._dag

  
def evaluate(node):
  if not isinstance(node, Expr):
    return node
  
  from . import backend
  return backend.evaluate(spartan.get_master(), dag(node))
     


class Op(Expr):
  def node_init(self):
    if self.children is None: self.children = tuple()
    if isinstance(self.children, list): self.children = tuple(self.children)
    if not isinstance(self.children, tuple): self.children = (self.children,)
    
    self.children = [lazify(c) for c in self.children]


class IndexExpr(Expr, Node):
  _members = ['src', 'idx']

class ReduceExtentsExpr(Op, Node):
  _members = ['children', 'axis', 'dtype_fn', 'local_reducer_fn', 'combiner_fn']
 
class MapTilesExpr(Op, Node):
  _members = ['children', 'map_fn', 'fn_kw']

class MapExtentsExpr(Op, Node):
  _members = ['children', 'map_fn', 'fn_kw']

class OuterProductExpr(Op, Node):
  _members = ['children', 'map_fn', 'map_fn_kw', 'reduce_fn', 'reduce_fn_kw']
  
class NdArrayExpr(Expr, Node):
  _members = ['_shape', 'dtype', 'tile_hint']
  
class StencilExpr(Expr, Node):
  _members = ['images', 'filters', 'stride']
  
def stencil(image, filters, stride=1):
  return StencilExpr(image, filters, stride)


def map_extents(v, fn, shape_hint=None, **kw):
  '''
  Evaluate ``fn`` over each extent of the input.
  
  ``fn`` should take (extent, [input_list], **kw)
  
  :param v:
  :param fn:
  '''
  return MapExtentsExpr(v, map_fn=fn, fn_kw=kw)


def map_tiles(v, fn, **kw):
  '''
  Evaluate ``fn`` over each tile of the input.
  
  ``fn`` should be of the form ([inputs], **kw).
  :param v:
  :param fn:
  '''
  return MapTilesExpr(v, map_fn=fn, fn_kw=kw)


def ndarray(shape, dtype=np.float, tile_hint=None):
  '''
  Lazily create a new distributed array.
  :param shape:
  :param dtype:
  :param tile_hint:
  '''
  return NdArrayExpr(_shape = shape,
                     dtype = dtype,
                     tile_hint = tile_hint) 


def reduce_extents(v, axis,
                   dtype_fn,
                   local_reducer_fn,
                   combiner_fn):
  return ReduceExtentsExpr(v, axis, dtype_fn, local_reducer_fn, combiner_fn)


def outer_product(a, b, map_fn, reducer_fn):
  return OuterProductExpr(a, b, map_fn, reducer_fn)

def outer(a, b):
  return OuterProductExpr(a, b, map_fn=np.dot, reducer_fn=np.add)

def _sum_local(index, tile, axis):
  return np.sum(tile[:], axis)

def _sum_reducer(a, b):
  return a + b

def sum(x, axis=None):
  return reduce_extents(x, axis=axis,
                       dtype_fn = lambda input: input.dtype,
                       local_reducer_fn = _sum_local,
                       combiner_fn = lambda a, b: a + b)
    

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

#  util.log('Index for reduction: %s %s %s',
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
  x = x.evaluate()
  compute_min = reduce_extents(x, axis,
                               dtype_fn = _argmin_dtype,
                               local_reducer_fn = _argmin_local,
                               combiner_fn = _argmin_reducer)
  
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
  
  ravelled_ex = extent.TileExtent((ul,), (lr - ul,), shape)
  ravelled_data = inputs[0][ex].ravel()
  return ravelled_ex, ravelled_data
   
def ravel(v):
  return map_extents(v, _ravel_mapper)

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

  target_shape = (inputs[0].shape[1], inputs[1].shape[0])
  
  # fetch corresponding column tile of array 'b'
  # rows = ex_a.cols
  # cols = *
  ex_b = extent.TileExtent((ex_a.ul[1], 0),
                           (ex_a.lr[1] - ex_a.ul[1], inputs[1].shape[1]),
                           inputs[1].shape)
  b = inputs[1].fetch(ex_b)
  result = np.dot(a, b)
  out = extent.TileExtent([ex_a.ul[0], 0],
                          result.shape,
                          target_shape)
  
  return out, result

def _dot_numpy(inputs, ex, numpy_data=None):
  return (ex[0].add_dim(), np.dot(inputs[0][ex], numpy_data))
  

def dot(a, b):
  av = evaluate(a)
  bv = evaluate(b)
  
  if isinstance(bv, np.ndarray):
    return map_extents((av,), _dot_numpy, numpy_data=bv)
  
  av, bv = distarray.broadcast(av, bv)
  Assert.eq(a.shape[1], b.shape[0])
  return map_extents((av, bv), _dot_mapper)
            

def map(v, fn, axis=None, **kw):
  return map_tiles(v, fn, **kw)

def rand(*shape, **kw):
  '''
  :param tile_hint: A tuple indicating the desired tile shape for this array.
  '''
  return map_extents(ndarray(shape, 
                             dtype=np.float, 
                             tile_hint=kw.get('tile_hint', None)), 
                     fn = lambda inputs, ex: (ex, np.random.rand(*ex.shape)))
  
def randn(*shape, **kw):
  return map_extents(ndarray(shape, 
                             dtype=np.float, 
                             tile_hint=kw.get('tile_hint', None)), 
                     fn = lambda inputs, ex: (ex, np.random.randn(*ex.shape)))

def zeros(shape, dtype=np.float, tile_hint=None):
  return map_extents(ndarray(shape, dtype=np.float, tile_hint=tile_hint), 
                     fn = lambda inputs, ex: (ex, np.zeros(ex.shape, dtype)))

def ones(shape, dtype=np.float, tile_hint=None):
  return map_extents(ndarray(shape, dtype=np.float, tile_hint=tile_hint), 
                     fn = lambda inputs, ex: (ex, np.ones(ex.shape, dtype)))


def _arange_mapper(inputs, ex, dtype=None):
  pos = extent.ravelled_pos(ex.ul, ex.array_shape)
  #util.log('Extent: %s, pos: %s', ex, pos)
  sz = np.prod(ex.shape)
  return (ex, np.arange(pos, pos+sz, dtype=dtype).reshape(ex.shape))


def arange(shape, dtype=np.float):
  return map_extents(ndarray(shape, dtype=dtype), 
                     fn = _arange_mapper,
                     dtype=dtype)

