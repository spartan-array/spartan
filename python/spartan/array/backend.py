#!/usr/bin/env python

from . import prims
from spartan import util
from spartan.dense import distarray, extent, tile
from spartan.util import join_tuple, Assert, divup
import numpy as np
import math

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
  idx_vals = idx.fetch(extent.drop_axis(ex, -1))
  
  util.log_info('Dest shape: %s, idx: %s, %s', tile.shape, ex, idx_vals)
  for dst_idx, src_idx in enumerate(idx_vals):
    tile[dst_idx] = src.fetch(extent.from_slice(int(src_idx), src.shape))
  return [(ex, tile)]

def bool_index_mapper(ex, tile, src, idx):
  slc = ex.to_slice()
  local_val = src[slc]
  local_idx = idx[slc]
  return [(ex, local_val[local_idx])]


try:
  import parakeet
  jit = parakeet.jit
except:
  def jit(fn):
    return fn
  
def convolve(local_image, local_filters):
  num_images, w, h = local_image.shape
  num_filts, fw, fh = local_filters.shape

  def _inner(args):
    iid, fid, x, y = args
    image = local_image[iid]
    f = local_filters[fid]
    out = 0
    for i in xrange(fw):
      for j in xrange(fh):
        if x + i < w and y + j < h:
          out += image[x + i, y + j] * f[i, j]                                                                                                                            
    return out

  return parakeet.imap(_inner, (num_images, num_filts, w, h))


        
def stencil_mapper(region, local, filters=None, image=None, target=None):
  local_filters = filters.glom()
  local_image = image.fetch(region)
  
  num_img, w, h = image.shape
  num_filt, fw, fh = filters.shape
  
  util.log_info('Stencil(%s), image: %s, filter: %s (%s, %s)',
           region,
           local_image.shape, local_filters.shape,
           image.shape, filters.shape)
  
  target_region = extent.TileExtent(
      (region.ul[0], 0, region.ul[1], region.ul[2]),
      (region.sz[0], num_filt, region.sz[1], region.sz[2]),
      target.shape)

  result = convolve(local_image, local_filters)
  
  util.log_info('Updating: %s', target_region)
  target.update(target_region, result)
   
 
class Backend(object):
  def eval_Value(self, ctx, prim, inputs):
    return prim.value
  
  def eval_MapTiles(self, ctx, prim, inputs):
    largest = largest_value(inputs)
    inputs = distarray.broadcast(inputs)
    map_fn = prim.map_fn
    fn_kw = prim.fn_kw or {}
    
    util.log_info('Mapping %s over %d inputs; largest = %s', 
             map_fn, len(inputs), largest.shape)
    
    def mapper(ex, _):
      #util.log_info('MapTiles: %s', map_fn)
      #util.log_info('Fetching %d inputs', len(inputs))
      #util.log_info('%s %s', inputs, ex)
      local_values = [input.fetch(ex) for input in inputs]
      #util.log_info('Mapping...')
      result = map_fn(local_values,  **fn_kw)
      #util.log_info('Done.')
      assert isinstance(result, np.ndarray), result
      return [(ex, tile.from_data(result))]
    
    result = distarray.map_to_array(largest, mapper)
    return result
  
  def eval_MapExtents(self, ctx, prim, inputs):
    if prim.target is not None:
      inputs, target = inputs[:-1], inputs[-1]
    else:
      target = None
            
    map_fn = prim.map_fn
    reduce_fn = prim.reduce_fn
    fn_kw = prim.fn_kw or {}
    
    if target is not None:
      def mapper(ex, _):
        new_extent, result = map_fn(inputs, ex, **fn_kw)
        target.update(new_extent, result)
        
      inputs[0].foreach(mapper)
      return target
    else:
      def mapper(ex, _):
        #util.log_info('MapExtents: %s', map_fn)
        new_extent, result = map_fn(inputs, ex, **fn_kw)
        return [(new_extent, tile.from_data(result))]
      return distarray.map_to_array(inputs[0], 
                                    mapper_fn = mapper,
                                    reduce_fn = reduce_fn)
  
  
  def eval_NewArray(self, ctx, prim, inputs):
    shape = prim.array_shape
    dtype = prim.dtype
    tile_hint = prim.tile_hint
    
    if prim.combine_fn is not None:
      combiner = tile.TileAccum(prim.combine_fn)
    else:
      combiner = None
      
    if prim.reduce_fn is not None:
      reducer = tile.TileAccum(prim.reduce_fn)
    else:
      reducer = None
       
    return distarray.create(ctx, shape, dtype,
                            combiner=combiner,
                            reducer=reducer,
                            tile_hint=tile_hint)
  
  def eval_Reduce(self, ctx, prim, inputs):
    input_array = inputs[0]
    dtype = prim.dtype_fn(input_array)
    axis = prim.axis
    util.log_info('Reducing %s over axis %s', input_array.shape, prim.axis)
    shape = extent.shape_for_reduction(input_array.shape, prim.axis)
    tile_accum = tile.TileAccum(prim.combine_fn)
    output_array = distarray.create(ctx, shape, dtype,
                                    combiner=tile_accum, 
                                    reducer=tile_accum)
    local_reducer = prim.local_reduce_fn
    
    util.log_info('Reducing into array %d', output_array.table.id())
    
    def mapper(ex, tile):
      #util.log_info('Reduce: %s', local_reducer)
      reduced = local_reducer(ex, tile, axis)
      dst_extent = extent.index_for_reduction(ex, axis)
      output_array.update(dst_extent, reduced)
    
    input_array.foreach(mapper)
    
    return output_array
    
  def eval_Slice(self, ctx, prim, inputs):
    src = inputs[0]
    idx = inputs[1]
    
    return distarray.Slice(src, idx)
    
    slice_region = extent.from_slice(idx, src.shape)
    matching_extents = dict(extent.extents_for_region(src.extents, slice_region))
    
    util.log_info('Taking slice: %s from %s', idx, src.shape)
    #util.log_info('Matching: %s', matching_extents)
    result = distarray.map_to_array(
      src, lambda k, v: slice_mapper(k, v, slice_region, matching_extents))
    
    util.log_info('Done.')
    return result
  
  def eval_Index(self, ctx, prim, inputs):
    src = inputs[0]
    idx = inputs[1]
    
    Assert.isinstance(idx, (np.ndarray, distarray.DistArray))
    
    if idx.dtype == np.bool:
      dst = distarray.map_to_array(src, bool_index_mapper)
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
      dst.foreach(lambda k, v: int_index_mapper(k, v, src, idx, dst))
      return dst
    
    
  def eval_Stencil(self, ctx, prim, inputs):
    image = inputs[0]
    filters = inputs[1] 
    
    num_img, w, h = image.shape
    num_filt, fw, fh = filters.shape
    
    tile_size = util.divup(w, math.sqrt(ctx.num_workers())) 
    
    dst = distarray.empty(ctx, (num_img, num_filt, w, h), image.dtype,
                          reducer=distarray.accum_sum,
                          tile_hint=(num_img, num_filt, tile_size, tile_size))
                          
    image.foreach(lambda k, v: stencil_mapper(k, v, filters, image, dst))
    return dst
  
  def _evaluate(self, ctx, prim):
    inputs = [self.evaluate(ctx, v) for v in prim.dependencies()]
    #util.log_info('Evaluating: %s', prim.typename())
    return getattr(self, 'eval_' + prim.typename())(ctx, prim, inputs)    
      
  
  def evaluate(self, ctx, prim):
    #util.log_info('Evaluating: %s', prim)
    Assert.isinstance(prim, prims.Primitive) 
    if prim.cached_value is None:
      prim.cached_value = self._evaluate(ctx, prim)
    
    return prim.cached_value
  
  
def evaluate(ctx, prim):
  return Backend().evaluate(ctx, prim)