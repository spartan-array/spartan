'''
Indexing operations (slicing and filtering).
'''
from .base import Expr, LazyList
from spartan import util
from spartan.array import extent, tile, distarray
from spartan.node import Node
from spartan.util import Assert, join_tuple
import numpy as np


class IndexExpr(Expr):
  __metaclass__ = Node
  _members = ['src', 'idx']

  def node_init(self):
    Expr.node_init(self)
    assert not isinstance(self.src, LazyList)
    assert not isinstance(self.idx, LazyList)
    assert not isinstance(self.idx, list)

  def evaluate(self, ctx, deps):
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
  tile = src.fetch(ex)
  
  util.log_info('Dest shape: %s, idx: %s, %s', tile.shape, ex, idx_vals)
  for dst_idx, src_idx in enumerate(idx_vals):
    tile[dst_idx] = src.fetch(extent.from_slice(int(src_idx), src.shape))

  return [(ex, tile)]

def bool_index_mapper(ex, src, idx):
  val = src.fetch(ex)
  mask = idx.fetch(ex)

  #util.log_info('\nVal: %s\n Mask: %s', val, mask)
  return [(ex, np.ma.masked_array(val, mask))]

def eval_Index(ctx, prim, deps):
  src = deps['src']
  idx = deps['idx']
  
  Assert.isinstance(idx, (np.ndarray, distarray.DistArray))

  if idx.dtype == np.bool:
    # return a new array masked by `idx`
    dst = src.map_to_array(bool_index_mapper, kw={ 'src' : src, 'idx' : idx})
    return dst
  else:
    util.log_info('Integer indexing...')

    Assert.eq(len(idx.shape), 1)

    # create empty destination of same first dimension as the index array
    dst = distarray.create(join_tuple([idx.shape[0]], src.shape[1:]), dtype=src.dtype)
    
    # map over it, fetching the appropriate values for each tile.
    return dst.map_to_array(int_index_mapper, kw={ 'src' : src, 'idx' : idx, 'dst' : dst })

    
def eval_Slice(ctx, prim, deps):
  src = deps['src']
  idx = deps['idx']
  
  return distarray.Slice(src, idx)
  # return _eager_slice(src, idx)


