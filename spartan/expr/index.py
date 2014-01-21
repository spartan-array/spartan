'''
Indexing operations (slicing and filtering).
'''
import numpy as np
from spartan import util, blob_ctx
from spartan.array import extent, tile, distarray
from spartan.node import Node
from spartan.util import Assert, join_tuple

from .base import Expr, ListExpr
from .map import MapResult


class IndexExpr(Expr):
  __metaclass__ = Node
  _members = ['src', 'idx']

  def node_init(self):
    Expr.node_init(self)
    assert not isinstance(self.src, ListExpr)
    assert not isinstance(self.idx, ListExpr)
    assert not isinstance(self.idx, list)

  def _evaluate(self, ctx, deps):
    idx = deps['idx']
    assert not isinstance(idx, list) 
    util.log_info('Indexing: %s', idx)
    if isinstance(idx, tuple) or\
       isinstance(idx, slice) or\
       np.isscalar(idx):
      return eval_Slice(ctx, self, deps)
    
    return eval_Index(ctx, self, deps)
     

def int_index_mapper(ex, src, idx, dst):
  '''Map over the index array, fetching rows from the data array.'''
  idx_vals = idx.fetch(extent.drop_axis(ex, -1))

  output = []
  for dst_idx, src_idx in enumerate(idx_vals):
    output.append(src.select(src_idx))

  output_ex = extent.create(
    ([ex.ul[0]] + [0] * (len(dst.shape) - 1)),
    ([ex.lr[0]] + list(output[0].shape)),
    (dst.shape))

  #util.log_info('%s %s', output_ex.shape, np.array(output).shape)
  output_tile = tile.from_data(np.array(output))
  tile_id = blob_ctx.get().create(output_tile).wait().blob_id
  return MapResult([(output_ex, tile_id)], None)

def bool_index_mapper(ex, src, idx):
  val = src.fetch(ex)
  mask = idx.fetch(ex)

  #util.log_info('\nVal: %s\n Mask: %s', val, mask)
  masked_val = np.ma.masked_array(val, mask)
  output_tile = tile.from_data(masked_val)
  tile_id = blob_ctx.get().create(output_tile).wait().blob_id
  return MapResult([(ex, tile_id)], None)

def eval_Index(ctx, prim, deps):
  src = deps['src']
  idx = deps['idx']
  
  Assert.isinstance(idx, (np.ndarray, distarray.DistArray))

  if idx.dtype == np.bool:
    # return a new array masked by `idx`
    dst = distarray.map_to_array(src, bool_index_mapper, kw={ 'src' : src, 'idx' : idx})
    return dst
  else:
    util.log_info('Integer indexing...')

    Assert.eq(len(idx.shape), 1)

    # create empty destination of same first dimension as the index array
    dst = distarray.create(join_tuple([idx.shape[0]], src.shape[1:]), dtype=src.dtype)
    
    # map over it, fetching the appropriate values for each tile.
    return distarray.map_to_array(dst,
                                  int_index_mapper, kw={ 'src' : src, 'idx' : idx, 'dst' : dst })

    
def eval_Slice(ctx, prim, deps):
  src = deps['src']
  idx = deps['idx']
  
  return distarray.Slice(src, idx)
  # return _eager_slice(src, idx)


