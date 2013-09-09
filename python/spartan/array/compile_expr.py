#!/usr/bin/env python

'''Convert from numpy expression trees to the lower-level
operations supported by the backends (see `prims`).

'''

from . import expr, prims, distarray
from .. import util
from ..util import Assert
from spartan.config import flags
from .extent import index_for_reduction, shapes_match
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


def _to_structured_array(**kw):
  '''Create a structured array from the given input arrays.'''
  out = np.ndarray(kw.values()[0].shape, 
                  dtype=','.join([a.dtype.str for a in kw.itervalues()]))
  
  for k, v in kw.iteritems():
    out[k] = v
  return out



def _argmin_local(index, value, axis):
  local_idx = value.argmin(axis)
  local_min = value.min(axis)

#  util.log('Index for reduction: %s %s %s',
#           index.array_shape,
#           axis,
#           index_for_reduction(index, axis))

  global_idx = index.to_global(local_idx, axis)

  new_idx = index_for_reduction(index, axis)
  new_value = _to_structured_array(idx=global_idx, min=local_min)

#   print index, value.shape, axis
#   print local_idx.shape
  assert shapes_match(new_idx, new_value), (new_idx, new_value.shape)
  return [(new_idx, new_value)]

def _argmin_reducer(a, b):
  return np.where(a['min'] < b['min'], a, b)

def _sum_local(index, tile, axis):
  return np.sum(tile[:], axis)

def _sum_reducer(a, b):
  return a + b

def _apply_binary_op(inputs, binary_op=None):
  assert len(inputs) == 2
  return binary_op(*inputs)

class OpToPrim(object):
  def compile_index(self, op, children):
    src, idx = children
    
    # differentiate between slices (cheap) and index/boolean arrays (expensive)
    if isinstance(idx, prims.Value) and\
       (isinstance(idx.value, tuple) or 
        isinstance(idx.value, slice)):
      return prims.Slice(src, idx)
    else:
      return prims.Index(src, idx)
  
  
  
  def compile_sum(self, op, children):
    axis = op.kwargs.get('axis', None)
    return prims.Reduce(children[0],
                        axis,
                        dtype_fn = lambda input: input.dtype,
                        local_reducer_fn = lambda ex, v: _sum_local(ex, v, axis),
                        combiner_fn = lambda a, b: a + b)
    
  
  def compile_argmin(self, op, children):
    axis = op.kwargs.get('axis', None)
    compute_min = prims.Reduce(children[0],
                               axis,
                               dtype_fn = lambda input: 'i8,f8',
                               local_reducer_fn = _argmin_local,
                               combiner_fn = _argmin_reducer)
    
    def _take_idx_mapper(tile):
      return tile['idx']
    
    take_indices = prims.MapTiles([compute_min], _take_idx_mapper, fn_kw = {})
    
    return take_indices
  
  
  def compile_map_extents(self, op, children):
    Assert.eq(len(children), 1)
    child = children[0]
    return prims.MapExtents([child], 
                            map_fn = op.kwargs['map_fn'],
                            fn_kw = op.kwargs['fn_kw'])
                              
  
  
  def compile_map_tiles(self, op, children):
    Assert.eq(len(children), 1)
    child = children[0]
    return prims.MapTiles([child], 
                          map_fn = op.kwargs['map_fn'],
                          fn_kw = op.kwargs['fn_kw'])
   
    
  def compile_ndarray(self, op, children):
    shape = op.kwargs['shape']
    dtype = op.kwargs['dtype']
    return prims.NewArray(array_shape=shape, dtype=dtype)
    
  def compile_op(self, op):
    '''Convert a numpy expression tree in an Op tree.
    :param op:
    :rval: DAG of `Primitive` operations.
    '''
    if isinstance(op, expr.LazyVal):
      return prims.Value(op.val)
    else:
      children = [self.compile_op(c) for c in op.children]
    
    if op.op in BINARY_OPS:
      return prims.MapTiles(children, 
                            _apply_binary_op, 
                            fn_kw = { 'binary_op' : op.op })
    
    if isinstance(op.op, str):
      op_key = op.op
    else:
      op_key = op.op.__name__
    
    return getattr(self, 'compile_' + op_key)(op, children)



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
                          dtype = self.visit(op.dtype))
  
  def visit_Value(self, op):
    return prims.Value(op.value)
  
  def visit_Index(self, op):
    return prims.Index(self.visit(op.src), self.visit(op.idx))
  
  def visit_Slice(self, op):
    return prims.Index(self.visit(op.src), self.visit(op.idx))



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
        # evaluate these operations directly and use the result; we don't 
        # avoid creating a new array for these operations.
        inputs.append(v)
        map_fn = _take_first
        fn_kw = {}
      
      op_ed = len(inputs)
      fns.append( { 'fn' : map_fn, 'fn_kw' : fn_kw, 'range' : (op_st, op_ed) } ) 
    
    map_fn = op.map_fn
    map_kw = op.fn_kw
    util.log('Map function: %s, kw: %s', map_fn, map_kw)
    
    util.log('Created fold mapper with %d inputs', len(inputs))
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
  
  print dag
  dag = apply_pass(FoldNumexprPass, dag)
  print dag
  dag = apply_pass(FoldMapPass, dag)
  return dag
