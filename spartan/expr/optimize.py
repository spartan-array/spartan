#!/usr/bin/env python

'''
Optimizations over an expression graph.
'''

import numpy as np
from spartan.config import FLAGS, BoolFlag
from spartan.array import extent
from spartan.expr.reduce import ReduceExpr
from spartan.node import Node
from spartan.util import Assert, iterable

from .. import util
from .base import Expr, LazyVal, LazyList, AsArray
from .shuffle import ShuffleExpr
from .map import MapExpr
from .ndarray import NdArrayExpr
from spartan.array.distarray import broadcast

try:
  import numexpr
except:
  numexpr = None
  
# Local versions of expression graph operations.
# These are used to build up fusion operations.

class LocalOp(Expr):
  __metaclass__ = Node
  _cached_value = None
  def evaluate(self, ctx):
    deps = {}
    for k, v in self.dependencies():
      if not iterable(v): deps[k] = v.evaluate()
      else: deps[k] = [sv.evaluate() for sv in v]
    
    self._cached_value = self._evaluate(ctx, deps)
    return self._cached_value
      

class LVal(LocalOp):
  '''
  A local value (a constant, or an input argument).
  '''
  _members = ['source', 'id']
  
  def dependencies(self): return {}
   
  def _evaluate(self, ctx, deps):
    return ctx[self.source][self.id]


class LMap(LocalOp):
  _members = ['fn', 'inputs', 'kw']
  
  def _evaluate(self, ctx, deps):
    return deps['fn'](deps['inputs'], **deps['kw'])
    
class LReduce(LocalOp):
  _members = ['fn', 'inputs', 'kw']
  
  def _evaluate(self, ctx, deps):
    return deps['fn'](deps['inputs'], ctx['extent'], **deps['kw'])


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


def _dag_eval(inputs, dag, kw_inputs):
  '''
  Function for evaluated folded operations.
  
  Takes 
  :param inputs:
  :param dag:
  '''
  ctx = { 'kw' : kw_inputs,
          'inputs' : inputs }
  return dag.evaluate(ctx, {})
    

def _fold_mapper(*inputs,  **kw):
  '''Helper mapper function for folding.
  
  Runs each callable in `fns` on a number of the input tiles.
  
  :param inputs: Input values for folded mappers
  :param fns: A list of dictionaries containing { 'fn', 'fn_kw', 'range' }
  :param map_fn:
  :param map_kw:
  '''
  fns = kw['fns']
  map_fn = kw['map_fn']
  map_kw = kw['map_kw']
  results = []
  for fn_info in fns:
    st, ed = fn_info['range']
    fn = fn_info['fn']
    kw = fn_info['fn_kw']
    fn_inputs = inputs[st:ed]
    result = fn(*fn_inputs, **kw)
    assert isinstance(result, np.ndarray) or np.isscalar(result), (type(result), fn)
    results.append(result)
  
  # util.log_info('%s %s %s', map_fn, results, map_kw)
  if 'numpy_expr' in map_kw: del map_kw['numpy_expr']
  return map_fn(*results, **map_kw)
    

def _take_first(*lst):
#  util.log_info('Take first: %s --> %s', lst, lst[0])
  return lst[0]

def map_like(v):
  return isinstance(v, (MapExpr, ShuffleExpr, NdArrayExpr, LazyVal, AsArray))


class MapMapFusion(OptimizePass):
  '''Fold sequences of Map operations together.
  
  map(f, map(g, map(h, x))) -> map(f . g . h, x)
  '''
  
  name = 'map_fusion'
   
  def visit_MapExpr(self, op):
    #util.log_info('Map tiles: %s', op.children)
    Assert.iterable(op.children)
    #util.log_info('VISIT %d', id(op))
    #util.log_info('VISIT %s', op.children)
    map_children = self.visit(op.children)
    all_maps = np.all([map_like(v) for v in map_children])
    
    #util.log_info('Folding: %s %s', [type(v) for v in map_children], all_maps)
    if not all_maps:
      return op.visit(self)
    
    children = []
    fns = []
    for v in map_children:
      op_st = len(children)
      
      if isinstance(v, MapExpr):
        op_in = self.visit(v.children)
        children.extend(op_in)
        map_fn = v.map_fn
        fn_kw = v.fn_kw
      else:
        # evaluate these operations directly and use the result; 
        # we can use the input of these operations, but can't
        # avoid creating a new array.
        children.append(v)
        map_fn = _take_first
        fn_kw = {}
      
      op_ed = len(children)
      fns.append({ 'fn' : map_fn, 'fn_kw' : fn_kw, 'range' : (op_st, op_ed) }) 
    
    map_fn = op.map_fn
    map_kw = op.fn_kw
    # util.log_info('Map function: %s, kw: %s', map_fn, map_kw)
    
    # util.log_info('Created fold mapper with %d children', len(children))
    return MapExpr(children=LazyList(vals=children),
                   map_fn=_fold_mapper,
                   fn_kw={ 'fns' : fns,
                           'map_fn' : map_fn,
                           'map_kw' : map_kw })

def _folded_reduce(ex, tile, axis,
                   map_inputs, map_fn, map_kw,
                   reduce_fn, reduce_kw,):
  map_inputs = broadcast(map_inputs)
  local_values = [v.fetch(ex) for v in map_inputs]
  map_output = map_fn(*local_values, **map_kw)
  return reduce_fn(ex, map_output, axis, **reduce_kw)

class ReduceMapFusion(OptimizePass):
  name = 'reduce_fusion'
  
  def visit_ReduceExpr(self, op):
    array = self.visit(op.array)
    
    if isinstance(array, MapExpr):
      children = array.children
      Assert.isinstance(children, LazyList)
      return ReduceExpr(array=children[0],
                        axis=op.axis,
                        dtype_fn=op.dtype_fn,
                        reduce_fn=_folded_reduce,
                        combine_fn=op.combine_fn,
                        fn_kw={'map_kw' : array.fn_kw,
                               'map_fn' : array.map_fn,
                               'reduce_fn' : op.reduce_fn,
                               'reduce_kw' : op.fn_kw,
                               'map_inputs' : children })
    else:
      return op.visit(self)



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
