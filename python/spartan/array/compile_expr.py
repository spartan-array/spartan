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

binary_ops = set([np.add, np.subtract, np.multiply, np.divide, np.mod, np.power,
                  np.equal, np.less, np.less_equal, np.greater, np.greater_equal])


def to_structured_array(**kw):
  '''Create a structured array from the given input arrays.'''
  out = np.ndarray(kw.values()[0].shape, 
                  dtype=','.join([a.dtype.str for a in kw.itervalues()]))
  
  for k, v in kw.iteritems():
    out[k] = v
  return out

def argmin_local(index, value, axis):
  local_idx = value.argmin(axis)
  local_min = value.min(axis)

#  util.log('Index for reduction: %s %s %s',
#           index.array_shape,
#           axis,
#           index_for_reduction(index, axis))

  global_idx = index.to_global(local_idx, axis)

  new_idx = index_for_reduction(index, axis)
  new_value = to_structured_array(idx=global_idx, min=local_min)

#   print index, value.shape, axis
#   print local_idx.shape
  assert shapes_match(new_idx, new_value), (new_idx, new_value.shape)
  return [(new_idx, new_value)]

def argmin_reducer(a, b):
  return np.where(a['min'] < b['min'], a, b)

def sum_local(index, tile, axis):
  return np.sum(tile[:], axis)

def sum_reducer(a, b):
  return a + b

def binary_op(fn, inputs, kw):
  return fn(*inputs)


def compile_index(op, children):
  src, idx = children
  
  # differentiate between slices (cheap) and index/boolean arrays (expensive)
  if isinstance(idx, prims.Value) and\
     (isinstance(idx.value, tuple) or 
      isinstance(idx.value, slice)):
    return prims.Slice(src, idx)
  else:
    return prims.Index(src, idx)


def compile_sum(op, children):
  axis = op.kwargs.get('axis', None)
  return prims.Reduce(children[0],
                      axis,
                      dtype_fn = lambda input: input.dtype,
                      local_reducer_fn = lambda ex, v: sum_local(ex, v, axis),
                      combiner_fn = lambda a, b: a + b)
  

def compile_argmin(op, children):
  axis = op.kwargs.get('axis', None)
  compute_min = prims.Reduce(children[0],
                             axis,
                             dtype_fn = lambda input: 'i8,f8',
                             local_reducer_fn = argmin_local,
                             combiner_fn = argmin_reducer)
  take_indices = prims.MapTiles([compute_min], lambda tile: tile['idx'])
  
  return take_indices


def compile_map_extents(op, children):
  Assert.eq(len(children), 1)
  child = children[0]
  return prims.MapExtents([child], map_fn = op.kwargs['map_fn'])  


def compile_map_tiles(op, children):
  Assert.eq(len(children), 1)
  child = children[0]
  return prims.MapTiles([child], map_fn = op.kwargs['map_fn'])  
 
  
def compile_ndarray(op, children):
  shape = op.kwargs['shape']
  dtype = op.kwargs['dtype']
  return prims.NewArray(array_shape=shape, dtype=dtype)
  
  
def compile_op(op):
  '''Convert a numpy expression tree in an Op tree.
  :param op:
  :rval: DAG of `Primitive` operations.
  '''
  if isinstance(op, expr.LazyVal):
    return prims.Value(op.val)
  else:
    children = [compile_op(c) for c in op.children]
  
  if op.op in binary_ops:
    return prims.MapTiles(children, lambda a, b: op.op(a, b))
  
  if isinstance(op.op, str):
    return globals()['compile_' + op.op](op, children)
  else:
    return globals()['compile_' + op.op.__name__](op, children)

def optimize(dag):
  if not flags.optimization:
    util.log('Optimizations disabled')
    return dag
  
  if flags.folding:
    folder = FoldMapPass()
    dag = folder.visit(dag)
  else:
    util.log('Folding disabled')
  
  return dag

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
                            map_fn = op.map_fn)  
  
  def visit_MapTiles(self, op):
    return prims.MapTiles(inputs = [self.visit(v) for v in op.inputs],
                          map_fn = op.map_fn)
  
  def visit_NewArray(self, op):
    return prims.NewArray(array_shape = self.visit(op.shape()),
                          dtype = self.visit(op.dtype))
  
  def visit_Value(self, op):
    return prims.Value(op.value)
  
  def visit_Index(self, op):
    return prims.Index(self.visit(op.src), self.visit(op.idx))
  
  def visit_Slice(self, op):
    return prims.Index(self.visit(op.src), self.visit(op.idx))


class FoldMapPass(OptimizePass):
  def visit_MapTiles(self, op):
    map_inputs = [self.visit(v) for v in op.inputs]
    
    all_maps = np.all([isinstance(v, (prims.Map, prims.NewArray, prims.Value)) 
                       for v in map_inputs])
    if not all_maps:
      return super(FoldMapPass, self).visit_MapTiles(op)
    
    # construct a mapper function which first evaluates inputs locally: [i_0... i_n]
    inputs = []
    ranges = []
    fns = []
    for v in map_inputs:
      op_st = len(inputs)
      if isinstance(v, (prims.Value, prims.NewArray, prims.MapExtents)):
        inputs.append(v)
        map_fn = lambda v: v
      else:
        op_in = [self.visit(child) for child in v.inputs]
        inputs.extend(op_in)
        map_fn = v.map_fn
      
      op_ed = len(inputs)
      fns.append(map_fn)
      ranges.append((op_st, op_ed))
    
    map_fn = op.map_fn
    def fold_mapper(*local_tiles):
      inputs = []
      for fn, (st, ed) in zip(fns, ranges):
        inputs.append(fn(*local_tiles[st:ed]))
      return map_fn(*inputs)
    
    util.log('Created fold mapper with %d inputs', len(inputs))
    return prims.MapTiles(inputs=inputs, map_fn = fold_mapper)