'''Lazy arrays.

Expr operations are not performed immediately, but are set aside
and built into a control flow graph, which is then compiled
into a series of primitive array operations.
'''

from node import node_type
from prims import NotShapeable
from spartan import util
from spartan.array import extent, distarray
from spartan.util import Assert
import numpy as np
import spartan
import types

class Expr(object):
  _dag = None
  
  def __add__(self, other):
    return Op(op=np.add, children=(self, other), numexpr='+')

  def __sub__(self, other):
    return Op(op=np.subtract, children=(self, other), numexpr='-')

  def __mul__(self, other):
    return Op(op=np.multiply, children=(self, other), numexpr='*')

  def __mod__(self, other):
    return Op(op=np.mod, children=(self, other), numexpr='%')

  def __div__(self, other):
    return Op(op=np.divide, children=(self, other))

  def __eq__(self, other):
    return Op(op=np.equal, children=(self, other))

  def __ne__(self, other):
    return Op(op=np.not_equal, children=(self, other))

  def __lt__(self, other):
    return Op(op=np.less, children=(self, other))

  def __gt__(self, other):
    return Op(op=np.greater, children=(self, other))

  def __pow__(self, other):
    return Op(op=np.power, children=(self, other))

  def __getitem__(self, idx):
    return Op('index', children=(self, lazify(idx)))

  def __setitem__(self, k, val):
    raise Exception, '__setitem__ not supported.'
  
  @property
  def shape(self):
    try:
      dag = self.dag()
      return dag.shape()
    except NotShapeable:
      return self.evaluate().shape
  
  def dag(self):
    if self._dag is not None:
      return self._dag
    
    from . import compile_expr
    dag = compile_expr.compile(self)
    dag = compile_expr.optimize(dag)
    self._dag = dag
    return self._dag

  def evaluate(self):
    from . import backend
    return backend.evaluate(spartan.get_master(), self.dag())
  
  def glom(self):
    return self.evaluate().glom()
    

Expr.__rsub__ = Expr.__sub__
Expr.__radd__ = Expr.__add__
Expr.__rmul__ = Expr.__mul__
Expr.__rdiv__ = Expr.__div__

@node_type
class LazyVal(Expr):
  _members = ['val']
  
  def __reduce__(self):
    return self.evaluate().__reduce__()


@node_type
class Op(Expr):
  _members = ['op', 'children', 'kwargs', 'numexpr']
  
  def node_init(self):
    if self.kwargs is None: self.kwargs = {}
    if self.children is None: self.children = tuple()
    
    Assert.isinstance(self.children, tuple)
    
    #util.log('%s', self.children)
    self.children = [lazify(c) for c in self.children]
  
  def _dtype(self):
    return self.args[0].dtype
  

def lazify(val):
  if isinstance(val, Expr): return val
  return LazyVal(val)

def val(x):
  return lazify(x)

def outer(a, b):
  return Op(np.outer, (a, b))

def sum(x, axis=None):
  return Op(np.sum, (x,), kwargs={'axis' : axis })

def argmin(x, axis=None):
  return Op(np.argmin, (x,), kwargs={'axis' : axis })

def size(x, axis=None):
  return Op('size', (x,), kwargs={'axis' : axis })

def mean(x, axis=None):
  return sum(x, axis) / size(x, axis)

def astype(x, dtype):
  assert x is not None
  return Op('astype', (x,), kwargs={ 'dtype' : dtype })

def ravel(v):
  return Op(np.ravel, (v,))

def diag(v):
  return Op(np.diag, (v,))

def diagflat(v):
  return diag(ravel(v))

Expr.outer = outer
Expr.sum = sum
Expr.mean = mean
Expr.astype = astype
Expr.diag = diag
Expr.diagflat = diagflat
Expr.ravel = ravel
Expr.argmin = argmin

def _dot_mapper(inputs, ex_a, ex_b):
  a = inputs[0].ensure(ex_a.to_slice())
  b = inputs[1].ensure(ex_b.to_slice())
  
  # fetch corresponding slice(s) of the other array
  result = np.dot(a, b)
  out = extent.TileExtent([ex_a.lr[0], ex_b.lr[1]],
                          result.shape,
                          (a.array_shape[0], b.array_shape[1]))
  return out, result


def dot(a, b):
  av = a.evaluate()
  bv = a.evaluate()
  av, bv = distarray.broadcast(av, bv)
  Assert.eq(a.shape[1], b.shape[0])
 
  return Op('map_extents', (a,), 
            kwargs={'map_fn' : _dot_mapper, 'fn_kw' : {} })
            

def map(v, fn, axis=None, **kw):
  if axis is None:
    return map_tiles(v, fn, **kw)
  
  return Op('map', (v,), 
            kwargs = {'map_fn' : fn, 'fn_kw' : kw, 'axis' : axis})


def map_extents(v, fn, **kw):
  '''
  Evaluate ``fn`` over each extent of the input.
  
  ``fn`` should take (extent, [input_list], **kw)
  :param v:
  :param fn:
  '''
  return Op('map_extents', (v,), kwargs={'map_fn' : fn, 'fn_kw' : kw})


def map_tiles(v, fn, **kw):
  '''
  Evaluate ``fn`` over each tile of the input.
  
  ``fn`` should be of the form ([inputs], **kw).
  :param v:
  :param fn:
  '''
  return Op('map_tiles', (v,), kwargs={'map_fn' : fn, 'fn_kw' : kw})


def ndarray(shape, dtype=np.float, tile_hint=None):
  '''
  Lazily create a new distribute array.
  :param shape:
  :param dtype:
  :param tile_hint:
  '''
  return Op('ndarray', 
            kwargs = { 'shape' : shape, 'dtype' : dtype, 'tile_hint' : tile_hint })


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
  pos = ex.ravelled_pos()
  util.log('Extent: %s, pos: %s', ex, pos)
  sz = np.prod(ex.shape)
  return (ex, np.arange(pos, pos+sz, dtype=dtype).reshape(ex.shape))

def arange(shape, dtype=np.float):
  return map_extents(ndarray(shape, dtype=dtype), 
                     fn = _arange_mapper,
                     dtype=dtype)

