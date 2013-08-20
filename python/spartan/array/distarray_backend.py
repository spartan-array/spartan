#!/usr/bin/env python

from . import prims, distarray
from spartan.pytable import util

def _evaluate(ctx, prim):
  if isinstance(prim, prims.Value):
    return prim.value
  elif isinstance(prim, prims.MapTiles):
    darray = prim.array
    mapper = prim.fn
    mapper_args = prim.args
    return darray.map_tiles(mapper, mapper_args)
  elif isinstance(prim, prims.NewArray):
    basis = prim.basis
    shape = prim.shape(basis)
    dtype = prim.dtype(basis)
    return distarray.DistArray.create(ctx, shape, dtype)
  elif isinstance(prim, prims.ReduceInto):
    src = prim.src
    dst = prim.dst
    reducer = prim.reducer
    reducer_args = prim.reducer_args
    dst.set_accumulator(reducer)
    
    

def evaluate(ctx, prim):
  if prim.cached_value is not None:
    return prim.cached_value
  else:
    prim.cached_value = _evaluate(prim)
  