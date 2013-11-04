'''
Indexing operations (slicing and filtering).
'''
from .base import Expr, LazyList
from spartan import util
from spartan.dense import extent, tile, distarray
from spartan.util import Assert, join_tuple
import numpy as np


class IndexExpr(Expr):
  _members = ['src', 'idx']

  def node_init(self):
    Expr.node_init(self)
    assert not isinstance(self.src, LazyList)
    assert not isinstance(self.idx, LazyList)
    assert not isinstance(self.idx, list)

  def evaluate(self, ctx, deps):
    idx = deps['idx']
    assert not isinstance(idx, list) 
    util.log_info('%s', idx)
    if isinstance(idx, tuple) or\
       isinstance(idx, slice) or\
       np.isscalar(idx):
      return eval_Slice(ctx, self, deps)
    
    return eval_Index(ctx, self, deps)
     

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

def eval_Index(ctx, prim, deps):
  src = deps['src']
  idx = deps['idx']
  
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
    dst = distarray.create(join_tuple([idx.shape[0]], src.shape[1:]),
                           dtype=src.dtype)
    
    # map over it, replacing existing items.
    dst.foreach(lambda k, v: int_index_mapper(k, v, src, idx, dst),
                kw={})
    return dst
  
    
def eval_Slice(ctx, prim, deps):
  src = deps['src']
  idx = deps['idx']
  
  return distarray.Slice(src, idx)
  # return _eager_slice(src, idx)

def _eager_slice(src, idx):  
  slice_region = extent.from_slice(idx, src.shape)
  matching_extents = dict(extent.find_overlapping(src.extents, slice_region))
  
  util.log_info('Taking slice: %s from %s', idx, src.shape)
  # util.log_info('Matching: %s', matching_extents)
  result = distarray.map_to_array(
    src, lambda k, v: slice_mapper(k, v, slice_region, matching_extents))
  
  util.log_info('Done.')
  return result
  
 
