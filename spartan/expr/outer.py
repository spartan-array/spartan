from .base import Expr
import numpy as np
from spartan.node import Node


class OuterProductExpr(Expr):
  __metaclass__ = Node
  _members = ['children', 'map_fn', 'map_fn_kw', 'reduce_fn', 'reduce_fn_kw']
  
def outer_product(a, b, map_fn, reduce_fn):
  '''
  Outer (cartesian) product over the tiles of ``a`` and ``b``.
  
  ``map_fn`` is applied to each pair; ``reduce_fn`` is used to 
  combine overlapping outputs.
  
  :param a:
  :param b:
  '''
  return OuterProductExpr(a, b, map_fn, reduce_fn)

def outer(a, b):
  
  return OuterProductExpr(a, b, map_fn=np.dot, 
                          reduce_fn=np.add)
