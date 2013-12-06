#!/usr/bin/env python

'''
Evalution for `Expr` nodes.

`evaluate` evaluates a nodes dependencies, caching
results, then evaluates nodes themselves.
'''

from spartan import util
from spartan.util import Assert

def _evaluate(ctx, prim):
  #util.log_info('Evaluting deps for %s', prim)
  deps = {}
  for k, vs in prim.dependencies().iteritems():
    deps[k] = evaluate(ctx, vs)
      
  #util.log_info('Evaluting %s', prim.typename())
  return prim.evaluate(ctx, deps)
  #return util.timeit(lambda: prim.evaluate(ctx, deps), 'eval: %s' % prim)

def evaluate(ctx, prim):
  '''
  Evaluate an `Expr`.  
 
  Dependencies are evaluated prior to evaluating ``prim``.
   
  Dependencies may be either a list, a dictionary or a single
  value of type `Expr`.  For convenience, we allow specifying 
  non-Expr dependencies; these are left unaltered.
  
  :param ctx:
  :param prim: `Expr`
  '''
  from .base import Expr
  
  #util.log_info('%s', type(prim))
  if isinstance(prim, dict):
    return dict([(k, evaluate(ctx, v)) for (k, v) in prim.iteritems()])
  elif isinstance(prim, tuple):
    return tuple([evaluate(ctx, v) for v in prim])
  elif isinstance(prim, list):
    return [evaluate(ctx, v) for v in prim]
  elif not isinstance(prim, Expr):
    return prim
  
  if prim._cached_value is None:
    #util.log_info('Evaluating: %s', prim)
    prim._cached_value = _evaluate(ctx, prim)
  
  return prim._cached_value

