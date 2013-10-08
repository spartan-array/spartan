#!/usr/bin/env python

'''Convert from numpy expression trees to the lower-level
operations supported by the backends (see `prims`).

'''

from . import expr, prims
from .. import util
from spartan.config import flags
import numpy as np

try:
  import numexpr
except:
  numexpr = None

# mapping from numpy functions to arithmetic operators
# this is used for numexpr folding
BINARY_OPS = { np.add : '+', 
               np.subtract : '-', 
               np.multiply : '*', 
               np.divide : '/', 
               np.mod : '%',
               np.power : '**',
               np.equal : '==', 
               np.less : '<', 
               np.less_equal : '<=', 
               np.greater : '>', 
               np.greater_equal : '>=' }

class OpToPrim(object):
  def compile_IndexExpr(self, op):
    src = self.compile_op(op.src)
    idx = self.compile_op(op.idx)
   
    # differentiate between slices (cheap) and index/boolean arrays (expensive)
    if (isinstance(idx, prims.Value) and
       (isinstance(idx.value, tuple) or isinstance(idx.value, slice))
       ):
      return prims.Slice(src, idx)
    else:
      return prims.Index(src, idx)
    
  def compile_StencilExpr(self, op):
    images = self.compile_op(op.images)
    filters = self.compile_op(op.filters)
    
    return prims.Stencil(images, filters, op.stride)
    
  def compile_ReduceExtentsExpr(self, op):
    children = [self.compile_op(c) for c in op.children]
    
    assert len(children) == 1
    return prims.Reduce(children[0],
                        axis=op.axis,
                        dtype_fn = op.dtype_fn,
                        local_reducer_fn = op.local_reducer_fn,
                        combiner_fn = op.combiner_fn)
                        
  def compile_MapExtentsExpr(self, op):
    children = [self.compile_op(c) for c in op.children]
    return prims.MapExtents(children,
                            map_fn = op.map_fn,
                            fn_kw = op.fn_kw)
                              
  
  
  def compile_MapTilesExpr(self, op):
    children = [self.compile_op(c) for c in op.children]
    return prims.MapTiles(children, 
                          map_fn = op.map_fn,
                          fn_kw = op.fn_kw)
   
    
  def compile_NdArrayExpr(self, op):
    return prims.NewArray(array_shape=op.shape, 
                          dtype=op.dtype, 
                          tile_hint=op.tile_hint)
    
  def compile_op(self, op):
    '''Convert a numpy expression tree in an Op tree.
    :param op:
    :rval: DAG of `Primitive` operations.
    '''
    if isinstance(op, expr.LazyVal):
      return prims.Value(op.val)
    
    return getattr(self, 'compile_' + op.node_type())(op)



class OptimizePass(object):
  def visit(self, op):
    if isinstance(op, prims.Primitive):
      return getattr(self, 'visit_%s' % op.node_type())(op)
    return op
  
  def visit_Reduce(self, op):
    return prims.Reduce(input = self.visit(op.input),
                        axis = op.axis,
                        dtype_fn = op.dtype_fn,
                        local_reducer_fn = op.local_reducer_fn,
                        combiner_fn = op.combiner_fn)
                        
  def visit_MapExtents(self, op):
    return prims.MapExtents(inputs = [self.visit(v) for v in op.inputs],
                            map_fn = op.map_fn,
                            fn_kw = op.fn_kw) 
  
  def visit_MapTiles(self, op):
    return prims.MapTiles(inputs = [self.visit(v) for v in op.inputs],
                          map_fn = op.map_fn,
                          fn_kw = op.fn_kw) 
  
  def visit_NewArray(self, op):
    return prims.NewArray(array_shape = self.visit(op.shape()),
                          dtype = self.visit(op.dtype),
                          tile_hint = op.tile_hint)
  
  def visit_Value(self, op):
    return prims.Value(op.value)
  
  def visit_Index(self, op):
    return prims.Index(self.visit(op.src), self.visit(op.idx))
  
  def visit_Slice(self, op):
    return prims.Slice(self.visit(op.src), self.visit(op.idx))
  
  def visit_Stencil(self, op):
    return prims.Stencil(self.visit(op.images),
                         self.visit(op.filters),
                         op.stride)



def _fold_mapper(inputs, fns=None, map_fn=None, map_kw=None):
  '''Helper mapper function for folding.
  
  Runs each fn in `fns` on a number of the input tiles.
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
  
  #util.log('%s %s %s', map_fn, results, map_kw)
  return map_fn(results, **map_kw)
    

def _take_first(lst): 
  return lst[0]

def map_like(v):
  return isinstance(v, (prims.Map, prims.NewArray, prims.Value))

class FoldMapPass(OptimizePass):
  '''Fold sequences of Map operations together.
  
  map(f, map(g, map(h, x))) -> map(f . g . h, x)
  '''
  
  name = 'fold'
   
  def visit_MapTiles(self, op):
    map_inputs = [self.visit(v) for v in op.inputs]
    all_maps = np.all([map_like(v) for v in map_inputs])
    
    if not all_maps:
      return super(FoldMapPass, self).visit_MapTiles(op)
    
    inputs = []
    fns = []
    for v in map_inputs:
      op_st = len(inputs)
      
      if isinstance(v, prims.MapTiles):
        op_in = [self.visit(child) for child in v.inputs]
        inputs.extend(op_in)
        map_fn = v.map_fn
        fn_kw = v.fn_kw
      else:
        # evaluate these operations directly and use the result; 
        # we can use the input of these operations, but can't
        # avoid creating a new array.
        inputs.append(v)
        map_fn = _take_first
        fn_kw = {}
      
      op_ed = len(inputs)
      fns.append( { 'fn' : map_fn, 'fn_kw' : fn_kw, 'range' : (op_st, op_ed) } ) 
    
    map_fn = op.map_fn
    map_kw = op.fn_kw
    #util.log('Map function: %s, kw: %s', map_fn, map_kw)
    
    #util.log('Created fold mapper with %d inputs', len(inputs))
    return prims.MapTiles(inputs=inputs, 
                          map_fn = _fold_mapper,
                          fn_kw = { 'fns' : fns,
                                    'map_fn' : map_fn,
                                    'map_kw' : map_kw })
  

def _numexpr_mapper(inputs, var_map=None, expr=None):
  gdict = {}
  for k, v in var_map.iteritems():
    gdict[k] = inputs[v]
    
  result = numexpr.evaluate(expr, global_dict = gdict)
  return result


_COUNTER = iter(xrange(1000000))
def new_var():
  return 'input_%d' % _COUNTER.next()
    

class FoldNumexprPass(OptimizePass):
  '''Fold binary operations compatible with numexpr into a single numexpr operator.'''
  name = 'numexpr'
  
  def visit_MapTiles(self, op):
    map_inputs = [self.visit(v) for v in op.inputs]
    all_maps = np.all([map_like(v) for v in map_inputs])
    
    if (not all_maps or 
        len(map_inputs) > 2 or
        op.map_fn != _apply_binary_op):
      return super(FoldNumexprPass, self).visit_MapTiles(op)
    
    a, b = map_inputs
    operation = op.fn_kw['binary_op']
   
    # mapping from variable name to input index 
    var_map = {}
    
    # inputs to the expression
    inputs = []
    expr = []
    
    def _add_expr(child):
      # fold expression from the a mapper into this one.
      if isinstance(child, prims.MapTiles) and child.map_fn == _numexpr_mapper:
        for k, v in child.fn_kw['var_map'].iteritems():
          var_map[k] = len(inputs)
          inputs.append(child.inputs[v])
        expr.extend(['(' + child.fn_kw['expr'] + ')'])
      else:
        v = new_var()
        var_map[v] = len(inputs)
        inputs.append(child)
        expr.append(v)
    
    _add_expr(a)
    expr.append(BINARY_OPS[operation])
    _add_expr(b)
    
    expr = ' '.join(expr)
    
    return prims.MapTiles(inputs=inputs,
                          map_fn = _numexpr_mapper,
                          fn_kw = { 
                                    'expr' : expr,
                                    'var_map' : var_map,
                                  })

def apply_pass(klass, dag):
  if not getattr(flags, 'opt_' + klass.name):
    util.log('Pass %s disabled', klass.name)
    return dag
  
  p = klass()
  return p.visit(dag)


def compile(expr):
  op_to_prim = OpToPrim()
  return op_to_prim.compile_op(expr)


def optimize(dag):
  if not flags.optimization:
    util.log('Optimizations disabled')
    return dag
  
  #print dag
  #dag = apply_pass(FoldNumexprPass, dag)
  dag = apply_pass(FoldMapPass, dag)
  #util.log('%s', dag)
  return dag
