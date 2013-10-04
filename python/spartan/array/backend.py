#!/usr/bin/env python

from . import prims
from spartan import util
from spartan.dense import distarray, extent, tile
from spartan.util import join_tuple, Assert
import numpy as np

def largest_value(vals):
  return sorted(vals, key=lambda v: np.prod(v.shape))[-1]

def slice_mapper(ex, val, region, matching_extents):
  if ex in matching_extents:
    intersection = matching_extents[ex]
    local_slc = extent.offset_slice(ex, intersection)
    output_ex = extent.offset_from(region, intersection)
    return [(output_ex, tile.from_data(val[local_slc]))]

def int_index_mapper(ex, tile, src, idx, dst):
  '''Map over the index array, fetching rows from the data array.'''
  idx_slc = ex.to_slice()[0]
  idx_vals = idx[idx_slc]
  
  util.log('Dest shape: %s, idx: %s, %s', tile.shape, ex, idx_vals)
  for dst_idx, src_idx in enumerate(idx_vals):
    tile[dst_idx] = src[int(src_idx)]
  return [(ex, tile)]

def bool_index_mapper(ex, tile, src, idx):
  slc = ex.to_slice()
  local_val = src[slc]
  local_idx = idx[slc]
  return [(ex, local_val[local_idx])]
  
    
class Backend(object):
  def eval_Value(self, ctx, prim, inputs):
    return prim.value
  
  def eval_MapTiles(self, ctx, prim, inputs):
    largest = largest_value(inputs)
    map_fn = prim.map_fn
    fn_kw = prim.fn_kw or {}
    
    util.log('Mapping over %d inputs; largest = %s', len(inputs), largest.shape)
    
    def mapper(ex, _):
      util.log('MapTiles: %s', map_fn)
      slc = ex.to_slice()
      #util.log('Fetching %d inputs', len(inputs))
      local_values = [input[slc] for input in inputs]
      #util.log('Mapping...')
      result = map_fn(local_values,  **fn_kw)
      #util.log('Done.')
      assert isinstance(result, np.ndarray), result
      return [(ex, tile.from_data(result))]
    
    result = largest.map_to_array(mapper)
    return result
  
  def eval_MapExtents(self, ctx, prim, inputs):
    map_fn = prim.map_fn
    fn_kw = prim.fn_kw or {}
    
    def mapper(ex, _):
      #util.log('MapExtents: %s', map_fn)
      new_extent, result = map_fn(inputs, ex, **fn_kw)
      # util.log('MapExtents: %s, %s', ex, new_extent)
      return [(new_extent, tile.from_data(result))]
    
    return inputs[0].map_to_array(mapper)
  
  
  def eval_NewArray(self, ctx, prim, inputs):
    shape = prim.array_shape
    dtype = prim.dtype
    tile_hint = prim.tile_hint
    
    return distarray.create(ctx, shape, dtype, tile_hint=tile_hint)
  
  def eval_Reduce(self, ctx, prim, inputs):
    input_array = inputs[0]
    dtype = prim.dtype_fn(input_array)
    axis = prim.axis
    util.log('Reducing %s over axis %s', input_array.shape, prim.axis)
    shape = extent.shape_for_reduction(input_array.shape, prim.axis)
    tile_accum = tile.TileAccum(prim.combiner_fn)
    output_array = distarray.create(ctx, shape, dtype, reducer=tile_accum)
    local_reducer = prim.local_reducer_fn
    
    util.log('Reducing into array %d', output_array.table.id())
    
    def mapper(ex, tile):
      util.log('Reduce: %s', local_reducer)
      reduced = local_reducer(ex, tile, axis)
      dst_extent = extent.index_for_reduction(ex, axis)
      output_array.update(dst_extent, reduced)
    
    input_array.foreach(mapper)
    
    return output_array
    
  def eval_Slice(self, ctx, prim, inputs):
    src = inputs[0]
    idx = inputs[1]
    
    slice_region = extent.from_slice(idx, src.shape)
    matching_extents = dict(extent.extents_for_region(src.extents, slice_region))
    
    #util.log('Taking slice: %s from %s', idx, src.shape)
    #util.log('Matching: %s', matching_extents)
    
    return src.map_to_array(lambda k, v: slice_mapper(k, v, slice_region, matching_extents))
  
  def eval_Index(self, ctx, prim, inputs):
    dst = ctx.create_table()
    src = inputs[0]
    idx = inputs[1]
    
    Assert.eq(idx, (np.ndarray, distarray.DistArray))
    
    if idx.dtype == np.bool:
      dst = src.map_to_array(bool_index_mapper)
      # scan over output, compute running count of the size 
      # of the first dimension
      row_counts = src.map_to_table(lambda k, v: v.shape[0])
      for _, v in row_counts:
        pass
      raise NotImplementedError
    else:
      # create destination of the appropriate size
      dst = distarray.create(ctx, 
                             join_tuple([idx.shape[0]], src.shape[1:]),
                             dtype = src.dtype)
      
      # map over it, replacing existing items.
      return dst.map_inplace(lambda k, v: int_index_mapper(k, v, src, idx, dst))
  
  def _evaluate(self, ctx, prim):
    inputs = [self.evaluate(ctx, v) for v in prim.dependencies()]
    #util.log('Evaluating: %s', prim.typename())
    return getattr(self, 'eval_' + prim.typename())(ctx, prim, inputs)    
      
  
  def evaluate(self, ctx, prim):
    #util.log('Evaluating: %s', prim)
    Assert.isinstance(prim, prims.Primitive) 
    if prim.cached_value is None:
      prim.cached_value = self._evaluate(ctx, prim)
    
    return prim.cached_value
  
  
def evaluate(ctx, prim):
  return Backend().evaluate(ctx, prim)