#!/usr/bin/env python

'''Convert from numpy expression trees to the lower-level
operations supported by the backends (see `prims`).

'''

from . import expr, prims
from .. import util
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



def compile_op(op):
  '''Convert a numpy expression tree in an Op tree.'''
  if isinstance(op, expr.LazyVal):
    return prims.Value(op._val)
  else:
    children = [compile_op(c) for c in op.children]

  axis = op.kwargs.get('axis', None)
  
  if op.op in binary_ops:
    return prims.Map(children, 
                     lambda a, b: op.op(a, b))
  elif op.op == np.sum:
    return prims.Reduce(children[0],
                        axis,
                        dtype_fn = lambda input: input.dtype,
                        local_reducer_fn = lambda ex, v: sum_local(ex, v, axis),
                        combiner_fn = lambda a, b: a + b)
  elif op.op == np.argmin:
    compute_min = prims.Reduce(children[0],
                               axis,
                               dtype_fn = lambda input: 'i8,f8',
                               local_reducer_fn = argmin_local,
                               combiner_fn = argmin_reducer)
    take_indices = prims.Map(compute_min,
                             lambda tile: tile['idx'])
    return take_indices
  else:
    raise NotImplementedError, 'Compilation of %s not implemented yet.' % op.op
