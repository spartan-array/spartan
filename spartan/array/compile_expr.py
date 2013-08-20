#!/usr/bin/env python

'''Convert from numpy expression trees to the lower-level
operations supported by the backends (see `spartan.prims`).

'''

from . import expr, prims
from .. import util
import numpy as N


binary_ops = set([N.add, N.subtract, N.multiply, N.divide, N.mod, N.power,
                  N.equal, N.less, N.less_equal, N.greater, N.greater_equal])


def index_for_reduction(index, axis):
  return index.drop_axis(axis)

def shape_for_reduction(input_shape, axis):
  if axis == None: return (1,)
  input_shape = list(input_shape)
  del input_shape[axis]
  return input_shape

def shapes_match(offset, data):
  return N.all(offset.sz == data.shape)

def to_structured_array(*args):
  '''Create a structured array from the given input arrays.'''
  out = N.ndarray(args[0].shape, dtype=','.join([a.dtype.str for a in args]))
  for i, a in enumerate(args):
    out['f%d' % i] = a
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
  new_value = to_structured_array(global_idx, local_min)

#   print index, value.shape, axis
#   print local_idx.shape
  assert shapes_match(new_idx, new_value), (new_idx, new_value.shape)
  return [(new_idx, new_value)]

def argmin_reducer(a, b):
  return N.where(a['f1'] < b['f1'], a, b)

def sum_local(index, value, axis):
  return [(index_for_reduction(index, axis), N.sum(value, axis))]

def sum_reducer(a, b):
  return a + b

def binary_op(fn, inputs, kw):
  if kw is not None:
    return fn(*inputs, **kw)
  return fn(*inputs)

def compile_op(op):
  if isinstance(op, expr.LazyVal):
    return prims.Value(op._val)
  else:
    children = [compile_op(c) for c in op.children]

  if op.op in binary_ops:
    util.log(children)
    return prims.MapValues(prims.Join(children),
                           lambda inputs, kw: binary_op(op.op, inputs, kw),
                           op.kwargs)
  elif op.op == N.sum:
    tiled_sum = prims.MapTiles(
                      children[0], 
                      sum_local, 
                      op.kwargs['axis'])
    
    output = prims.NewArray(
                      basis=children[0],
                      shape=lambda basis: shape_for_reduction(basis.shape, op.kwargs['axis']),
                      dtype=lambda basis: basis.dtype)
    
    return prims.ReduceInto(src=tiled_sum, 
                            dst=output, 
                            reducer=sum_reducer, 
                            args=None)
  elif op.op == N.argmin:
    tiled_min = prims.MapTiles(
                      children[0], 
                      argmin_local, 
                      op.kwargs['axis'])
    output = prims.NewArray(
                      basis=children[0],
                      shape=lambda basis: shape_for_reduction(basis.shape, op.kwargs['axis']),
                      dtype=lambda basis: 'i8,f8')
    reduced = prims.ReduceInto(
                      src=tiled_min, 
                      dst=output, 
                      reducer=argmin_reducer, 
                      args=None)
    return prims.MapValues(reduced, lambda v, args: v['f0'], None)
  else:
    raise NotImplementedError, 'Compilation of %s not implemented yet.' % op.op
