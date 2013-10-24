#!/usr/bin/env python

'''
Optimizations over an expression graph.
'''

import numpy as np
from spartan.config import flags
from spartan.util import Assert

from .. import util
from .base import Expr, LazyVal, LazyList
from .map_extents import MapExtentsExpr
from .map_tiles import MapTilesExpr
from .ndarray import NdArrayExpr


try:
  import numexpr
except:
  numexpr = None

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
  

def _fold_mapper(inputs, fns=None, map_fn=None, map_kw=None):
  '''Helper mapper function for folding.
  
  Runs each callable in `fns` on a number of the input tiles.
  
  :param inputs: Input values for folded mappers
  :param fns: A list of dictionaries containing { 'fn', 'fn_kw', 'range' }
  :param map_fn:
  :param map_kw:
  '''
  results = []
  for fn_info in fns:
    st, ed = fn_info['range']
    fn = fn_info['fn']
    kw = fn_info['fn_kw']
    result = fn(inputs[st:ed], **kw)
    assert isinstance(result, np.ndarray), result
    results.append(result)
  
  # util.log_info('%s %s %s', map_fn, results, map_kw)
  return map_fn(results, **map_kw)
    

def _take_first(lst): 
  return lst[0]

def map_like(v):
  return isinstance(v, (MapTilesExpr, MapExtentsExpr, NdArrayExpr, LazyVal))

class FoldMapPass(OptimizePass):
  '''Fold sequences of Map operations together.
  
  map(f, map(g, map(h, x))) -> map(f . g . h, x)
  '''
  
  name = 'fold'
   
  def visit_MapTilesExpr(self, op):
    #util.log_info('Map tiles: %s', op.children)
    Assert.iterable(op.children)
    map_children = self.visit(op.children)
    all_maps = np.all([map_like(v) for v in map_children])
    
    #util.log_info('Folding: %s %s', [type(v) for v in map_children], all_maps)
    if not all_maps:
      return op.visit(self)
    
    children = []
    fns = []
    for v in map_children:
      op_st = len(children)
      
      if isinstance(v, MapTilesExpr):
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
    return MapTilesExpr(children=LazyList(vals=children),
                        map_fn=_fold_mapper,
                        fn_kw={ 'fns' : fns,
                                'map_fn' : map_fn,
                                'map_kw' : map_kw })
  

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
    

class FoldNumexprPass(OptimizePass):
  '''Fold binary operations compatible with numexpr into a single numexpr operator.'''
  name = 'numexpr'
  
  def visit_MapTilesExpr(self, op):
    map_children = [self.visit(v) for v in op.children]
    all_maps = np.all([map_like(v) for v in map_children])
   
    if not (all_maps and 'numpy_expr' in op.fn_kw):
      return op.visit(self)
    
    a, b = map_children
    operation = op.fn_kw['numpy_expr']
   
    # mapping from variable name to input index 
    var_map = {}
    
    # inputs to the expression
    inputs = []
    expr = []
    
    def _add_expr(child):
      # fold expression from the a mapper into this one.
      if isinstance(child, MapTilesExpr) and child.map_fn == _numexpr_mapper:
        for k, v in child.fn_kw['var_map'].iteritems():
          var_map[k] = len(inputs)
          inputs.append(child.children[v])
        expr.extend(['(' + child.fn_kw['numpy_expr'] + ')'])
      else:
        v = new_var()
        var_map[v] = len(inputs)
        inputs.append(child)
        expr.append(v)
    
    _add_expr(a)
    expr.append(operation)
    _add_expr(b)
    
    expr = ' '.join(expr)
    
    return MapTilesExpr(children=inputs,
                        map_fn=_numexpr_mapper,
                        fn_kw={ 'numpy_expr' : expr, 'var_map' : var_map, })

def apply_pass(klass, dag):
  if not getattr(flags, 'opt_' + klass.name):
    util.log_debug('Pass %s disabled', klass.name)
    return dag
  
  p = klass()
  return p.visit(dag)


def compile(expr):
  return expr

def optimize(dag):
  if not flags.optimization:
    util.log_info('Optimizations disabled')
    return dag
  
  if numexpr is not None:
    dag = apply_pass(FoldNumexprPass, dag)
  
  dag = apply_pass(FoldMapPass, dag)
  # util.log_info('%s', dag)
  return dag
