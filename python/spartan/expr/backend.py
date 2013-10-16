#!/usr/bin/env python

from spartan import util
from spartan.util import Assert

class Backend(object):
  def _evaluate(self, ctx, prim):
    deps = {}
    for k, vs in prim.dependencies().iteritems():
      assert isinstance(vs, list)
      deps[k] = [self.evaluate(ctx, v) for v in vs]
    
    return prim.evaluate(ctx, prim, deps)
  
  def evaluate(self, ctx, prim):
    from .base import Expr
    #util.log_info('Evaluating: %s', prim)
    Assert.isinstance(prim, Expr) 
    if prim._cached_value is None:
      prim._cached_value = self._evaluate(ctx, prim)
    
    return prim._cached_value
  
  
def evaluate(ctx, prim):
  return Backend().evaluate(ctx, prim)