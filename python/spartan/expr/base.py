'''Lazy arrays.

Expr operations are not performed immediately, but are set aside
and built into a control flow graph, which is then compiled
into a series of primitive array operations.
'''

from .node import Node
import numpy as np
import spartan

def _apply_binary_op(inputs, binary_op=None, numpy_expr=None):
  assert len(inputs) == 2
  return binary_op(*inputs)


class NotShapeable(Exception):
  pass


class Expr(object):
  _dag = None
  _cached_value = None
    
  def typename(self):
    return self.__class__.__name__
    
  
  def __add__(self, other):
    from .map_tiles import map_tiles
    return map_tiles((self, other), _apply_binary_op, binary_op=np.add, numpy_expr='+')

  def __sub__(self, other):
    from .map_tiles import map_tiles
    return map_tiles((self, other), _apply_binary_op, binary_op=np.subtract, numpy_expr='-')

  def __mul__(self, other):
    from .map_tiles import map_tiles
    return map_tiles((self, other), _apply_binary_op, binary_op=np.multiply, numpy_expr='*')

  def __mod__(self, other):
    from .map_tiles import map_tiles
    return map_tiles((self, other), _apply_binary_op, binary_op=np.mod, numpy_expr='%')

  def __div__(self, other):
    from .map_tiles import map_tiles
    return map_tiles((self, other), _apply_binary_op, binary_op=np.divide, numpy_expr='/')

  def __eq__(self, other):
    from .map_tiles import map_tiles
    return map_tiles((self, other), _apply_binary_op, binary_op=np.equal, numpy_expr='==')

  def __ne__(self, other):
    from .map_tiles import map_tiles
    return map_tiles((self, other), _apply_binary_op, binary_op=np.not_equal, numpy_expr='!=')

  def __lt__(self, other):
    from .map_tiles import map_tiles
    return map_tiles((self, other), _apply_binary_op, binary_op=np.less, numpy_expr='<')

  def __gt__(self, other):
    from .map_tiles import map_tiles
    return map_tiles((self, other), _apply_binary_op, binary_op=np.greater, numpy_expr='>')

  def __pow__(self, other):
    from .map_tiles import map_tiles
    return map_tiles((self, other), _apply_binary_op, binary_op=np.power, numpy_expr='**')

  def __getitem__(self, idx):
    from .index import IndexExpr
    return IndexExpr(src=self, idx=lazify(idx))

  def __setitem__(self, k, val):
    raise Exception, 'Expressions are read-only.'
  
  @property
  def shape(self):
    '''Try to compute the shape of this DAG.
    
    If the value has been computed already this always succeeds.
    '''
    if self._cached_value is not None:
      return self._cached_value.shape
    
    try:
      return self.compute_shape()
    except NotShapeable:
      return evaluate(self).shape
  
  def force(self):
    return force(self)
  
  def dag(self):
    return dag(self)
  
  def glom(self):
    return glom(self)

Expr.__rsub__ = Expr.__sub__
Expr.__radd__ = Expr.__add__
Expr.__rmul__ = Expr.__mul__
Expr.__rdiv__ = Expr.__div__


class LazyVal(Expr, Node):
  _members = ['val']
  
  def visit(self, visitor):
    return self
  
  def dependencies(self):
    return {}
  
  def compute_shape(self):
    return self.val.shape
  
  def evaluate(self, ctx, deps):
    return self.val
   
  def __reduce__(self):
    return evaluate(self).__reduce__()

def eval_LazyVal(ctx, prim, deps):
  return self.val

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
  '''
  Compile and return the DAG representing this expression.
  :param node:
  '''
  if not isinstance(node, Expr):
    raise TypeError
  
  if node._dag is not None:
    return node._dag
  
  from . import optimize
  dag = optimize.optimize(node)
  node._dag = dag
  return node._dag

  
def evaluate(node):
  '''
  Evaluate this expression.
  
  :param node:
  '''
  if not isinstance(node, Expr):
    return node
  
  from . import backend
  result = backend.evaluate(spartan.get_master(), dag(node))
  node._cached_value = result
  return result

force = evaluate

def eager(node):
  '''
  Eagerly evaluate ``node``.
  
  Convert the result back into an `Expr`.
  :param node: `Expr` to evaluate.
  '''
  return lazify(force(node))
  

def lazify(val):
  '''
  Lift ``val`` into an Expr node.
 
  If ``val`` is already an expression, it is returned unmodified.
   
  :param val:
  '''
  if isinstance(val, Expr): return val
  #util.log_info('Lazifying... %s', val)
  return LazyVal(val)


def val(x):
  return lazify(x)

  
class Op(Expr):
  def node_init(self):
    if self.children is None: self.children = tuple()
    if isinstance(self.children, list): self.children = tuple(self.children)
    if not isinstance(self.children, tuple): self.children = (self.children,)
    
    self.children = [lazify(c) for c in self.children]
