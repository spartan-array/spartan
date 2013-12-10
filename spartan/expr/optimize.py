#!/usr/bin/env python

'''
Optimizations over an expression graph.
'''

import numpy as np
from spartan.config import FLAGS, BoolFlag
from spartan.expr.reduce import ReduceExpr, ReduceOp
from spartan.util import Assert

from .. import util
from .base import Expr, Val, ListExpr, AsArray, Op, OpInput, make_var, DictExpr
from .shuffle import ShuffleExpr
from .map import MapExpr, MapOp
from .ndarray import NdArrayExpr
from spartan.array.distarray import broadcast

try:
  import numexpr
except:
  numexpr = None

try:
  import parakeet
except:
  parakeet = None


class OptimizePass(object):
  def __init__(self):
    self.visited = {}
  
  def visit(self, op):
    if not isinstance(op, Expr):
      return op
    
    if op in self.visited:
      return self.visited[op]
    
    #assert not op in self.visited, 'Infinite recursion during optimization %s' % op
    #self.visited.add(op)
    
    #util.log_info('VISIT %s: %s', op.typename(), hash(op))
    if hasattr(self, 'visit_%s' % op.typename()):
      self.visited[op] = getattr(self, 'visit_%s' % op.typename())(op)
    else:
      self.visited[op] = op.visit(self)
      
    return self.visited[op]

def map_like(v):
  return isinstance(v, (MapExpr, ShuffleExpr, NdArrayExpr, Val, AsArray))

def merge_var(children, k, v):
  if k in children:
    Assert.eq(v, children[k])
  else:
    children[k] = v


class MapMapFusion(OptimizePass):
  '''Fold sequences of Map operations together.
  
  map(f, map(g, map(h, x))) -> map(f . g . h, x)
  '''
  
  name = 'map_fusion'

  def visit_MapExpr(self, expr):
    #util.log_info('Map tiles: %s', expr)
    Assert.iterable(expr.children)
    #util.log_info('VISIT %d', id(op))
    #util.log_info('VISIT %s', op.children)
    map_children = self.visit(expr.children)
    all_maps = True
    Assert.isinstance(map_children, DictExpr)
    for k, v in map_children.iteritems():
      if not map_like(v):
        all_maps = False
        break

    if not all_maps:
      return expr.visit(self)

    #util.log_info('Original: %s', expr.op)
    children = {}
    combined_op = MapOp(fn=expr.op.fn,
                     kw=expr.op.kw,
                     pretty_fn=expr.op.pretty_fn)
    for name, child_expr in map_children.iteritems():
      if isinstance(child_expr, MapExpr):
        for k, v in child_expr.children.iteritems():
          merge_var(children, k, v)

        #util.log_info('Merging: %s', child_expr.op)
        combined_op.add_dep(child_expr.op)
      else:
        key = make_var()
        combined_op.add_dep(OpInput(idx=key))
        children[key] = child_expr

    return MapExpr(children=DictExpr(children),
                   op=combined_op)


class ReduceMapFusion(OptimizePass):
  name = 'reduce_fusion'
  
  def visit_ReduceExpr(self, expr):
    Assert.isinstance(expr.children, DictExpr)
    old_children = self.visit(expr.children)

    for k, v in old_children.iteritems():
      if not isinstance(v, MapExpr):
        return expr.visit(self)

    combined_op = ReduceOp(fn=expr.op.fn,
                           kw=expr.op.kw)

    new_children = {}
    for name, child_expr in old_children.iteritems():
      for k, v in child_expr.children.iteritems():
        merge_var(new_children, k, v)
      combined_op.add_dep(child_expr.op)

    return ReduceExpr(children=new_children,
                      axis=expr.axis,
                      dtype_fn=expr.dtype_fn,
                      combine_fn=expr.combine_fn,
                      op=combined_op)

def _numexpr_mapper(inputs, var_map=None, numpy_expr=None):
  gdict = {}
  for k, v in var_map.iteritems():
    gdict[k] = inputs[v]
    
  numexpr.ncores = 1 
  result = numexpr.evaluate(numpy_expr, global_dict=gdict)
  return result


_COUNTER = iter(xrange(1000000))
def new_var():
  return 'input_%d' % _COUNTER.next()
    

class NumexprFusionPass(OptimizePass):
  '''Fold binary operations compatible with numexpr into a single numexpr operator.'''
  name = 'numexpr_fusion'
  
  def visit_MapExpr(self, op):
    map_children = [self.visit(v) for v in op.children]
    all_maps = np.all([map_like(v) for v in map_children])
   
    if not (all_maps and op.numpy_expr is not None):
      return op.visit(self)
    
    a, b = map_children
    operation = op.numpy_expr
   
    # mapping from variable name to input index 
    var_map = {}
    
    # inputs to the expression
    inputs = []
    expr = []
    
    def _add_expr(child):
      # fold expression from the a mapper into this one.
      if isinstance(child, MapExpr) and child.map_fn == _numexpr_mapper:
        for k, v in child.fn_kw['var_map'].iteritems():
          var_map[k] = len(inputs)
          inputs.append(child.children[v])
        expr.extend(['(' + child.numpy_expr + ')'])
      else:
        v = new_var()
        var_map[v] = len(inputs)
        inputs.append(child)
        expr.append(v)
    
    _add_expr(a)
    expr.append(operation)
    _add_expr(b)
    
    expr = ' '.join(expr)
    
    return MapExpr(children=inputs,
                   map_fn=_numexpr_mapper,
                   numpy_expr = expr,
                   fn_kw={ 'numpy_expr' : expr, 'var_map' : var_map, })


def apply_pass(klass, dag):
  if not getattr(FLAGS, 'opt_' + klass.name):
    util.log_debug('Pass %s disabled', klass.name)
    return dag
  
  p = klass()
  return p.visit(dag)

passes = []

def optimize(dag):
  if not FLAGS.optimization:
    util.log_info('Optimizations disabled')
    return dag
  
  for p in passes:
    dag = apply_pass(p, dag)
  
  return dag

def add_optimization(klass, default):
  passes.append(klass)
  
  flagname = 'opt_' + klass.name
  #setattr(Flags, flagname, add_bool_flag(flagname, default=default))
  FLAGS.add(BoolFlag(flagname, default=default, help='Enable %s optimization' % klass.__name__))

add_optimization(MapMapFusion, True)
add_optimization(ReduceMapFusion, True)
add_optimization(NumexprFusionPass, False)

FLAGS.add(BoolFlag('optimization', default=True))
