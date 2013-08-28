'''Lazy arrays.

Expr operations are not performed immediately, but are set aside
and built into a control flow graph, which is then compiled
into a series of primitive array operations.
'''

from spartan import util
from spartan.node import Node
from spartan.util import Assert
import numpy as np
import spartan
import types
from spartan.array.prims import NotShapeable

class Expr(Node):
  _dag = None
  
  def __add__(self, other):
    return Op(op=np.add, children=(self, other))

  def __sub__(self, other):
    return Op(op=np.subtract, children=(self, other))

  def __mul__(self, other):
    return Op(op=np.multiply, children=(self, other))

  def __mod__(self, other):
    return Op(op=np.mod, children=(self, other))

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
    return Op('index', children=(self, idx))

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
    dag = compile_expr.compile_op(self)
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

class LazyVal(Expr):
  _members = ['val']

def lazify(val):
  if isinstance(val, Expr): return val
  return LazyVal(val)

def val(x):
  return lazify(x)

def pretty(op):
  if isinstance(op, (types.FunctionType,
                     types.BuiltinFunctionType,
                     types.MethodType,
                     types.BuiltinMethodType)):
    return op.__name__
  return repr(op)

class Op(Expr):
  _members = ['op', 'children', 'kwargs']
  
  def node_init(self):
    if self.kwargs is None: self.kwargs = {}
    if self.children is None: self.children = tuple()
    
    Assert.isinstance(self.children, tuple)
    
    #util.log('%s', self.children)
    self.children = [lazify(c) for c in self.children]
  
  def _dtype(self):
    return self.args[0].dtype

def outer(a, b):
  return Op(np.outer, (a, b))

def dot(a, b):
  return Op(np.dot, (a, b))

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
Expr.dot = dot
Expr.sum = sum
Expr.mean = mean
Expr.astype = astype
Expr.diag = diag
Expr.diagflat = diagflat
Expr.ravel = ravel
Expr.argmin = argmin


def map_extents(v, fn):
  return Op('map_extents', (v,), kwargs={'map_fn' : fn})

def map_tiles(v, fn):
  return Op('map_tiles', (v,), kwargs={'map_fn' : fn})

def ndarray(shape, dtype=np.float):
  return Op('ndarray', kwargs = { 'shape' : shape, 'dtype' : dtype })

def rand(*shape):
  return map_extents(ndarray(shape, dtype=np.float), 
                     fn = lambda inputs, ex: (ex, np.random.rand(*ex.shape)))
  
def randn(*shape):
  return map_extents(ndarray(shape, dtype=np.float), 
                     fn = lambda inputs, ex: (ex, np.random.randn(*ex.shape)))

def zeros(shape, dtype=np.float):
  return map_extents(ndarray(shape, dtype=np.float), 
                     fn = lambda inputs, ex: (ex, np.zeros(ex.shape, dtype)))

def ones(shape, dtype=np.float):
  return map_extents(ndarray(shape, dtype=np.float), 
                     fn = lambda inputs, ex: (ex, np.ones(ex.shape, dtype)))

def _arange_mapper(inputs, ex):
  pos = ex.ravelled_pos()
  sz = np.prod(ex.shape)
  return (ex, np.arange(pos, pos+sz).reshape(ex.shape))

def arange(shape, dtype=np.float):
  return map_extents(ndarray(shape, dtype=np.float), 
                     fn = _arange_mapper)

