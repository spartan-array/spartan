from .base import Op
from .node import Node
from spartan import util
from spartan.dense import extent, tile, distarray

class ReduceExtentsExpr(Op, Node):
  _members = ['children', 'axis', 'dtype_fn', 'local_reduce_fn', 'combine_fn']
  
  def dependencies(self):
    return { 'children' : self.children }

  def evaluate(self, ctx, prim, deps):
    input_array = deps['children'][0]
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
 
def reduce_extents(v, axis,
                   dtype_fn,
                   local_reduce_fn,
                   combine_fn):
  return ReduceExtentsExpr([v], axis, dtype_fn, local_reduce_fn, combine_fn)
