#!/usr/bin/env python

from . import prims, distarray
from spartan.pytable import util
import numpy as np

def largest_value(vals):
  return sorted(vals, key=lambda v: np.prod(v.shape))[-1]

def eval_Value(ctx, prim):
  return prim.value

def eval_Map(ctx, prim):
  inputs = [evaluate(ctx, v) for v in prim.inputs]
  largest = largest_value(inputs)
  map_fn = prim.map_fn
  
  #@util.trace_fn
  def mapper(ex, tile):
    slc = ex.to_slice()
    local_values = [input[slc] for input in inputs]
    result = map_fn(*local_values)
    assert isinstance(result, np.ndarray), result
    return [(ex, result)]
  
  return largest.map(mapper)


def eval_Reduce(ctx, prim):
  pass

def _evaluate(ctx, prim):
  return globals()['eval_' + prim.typename()](ctx, prim)    
    

def evaluate(ctx, prim):
  assert isinstance(prim, prims.Primitive), 'Not a primitive: %s' % prim
  util.log('Evaluating: %s', prim)
  if prim.cached_value is None:
    prim.cached_value = _evaluate(ctx, prim)
  
  return prim.cached_value    