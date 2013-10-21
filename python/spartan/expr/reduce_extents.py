from .base import Op
from .node import Node
from spartan import util
from spartan.dense import extent, tile, distarray

class ReduceExtentsExpr(Op, Node):
  _members = ['children', 'axis', 'dtype_fn', 'local_reduce_fn', 'combine_fn']
  
  def dependencies(self):
    return { 'children' : self.children }
  
  def visit(self, visitor):
    return ReduceExtentsExpr(
                        children=[visitor.visit(v) for v in self.children],
                        axis=self.axis,
                        dtype_fn=self.dtype_fn,
                        local_reduce_fn=self.local_reduce_fn,
                        combine_fn=self.combine_fn)
    

  def evaluate(self, ctx, deps):
    input_array = deps['children'][0]
    dtype = self.dtype_fn(input_array)
    axis = self.axis
    util.log_info('Reducing %s over axis %s', input_array.shape, self.axis)
    shape = extent.shape_for_reduction(input_array.shape, self.axis)
    tile_accum = tile.TileAccum(self.combine_fn)
    output_array = distarray.create(ctx, shape, dtype,
                                    combiner=tile_accum, 
                                    reducer=tile_accum)
    local_reducer = self.local_reduce_fn
    
    util.log_info('Reducing into array %d', output_array.table.id())
    
    def mapper(ex, tile):
      #util.log_info('Reduce: %s', local_reducer)
      reduced = local_reducer(ex, tile, axis)
      dst_extent = extent.index_for_reduction(ex, axis)
      util.log_info('Update: %s %s', dst_extent, reduced)
      output_array.update(dst_extent, reduced)
    
    input_array.foreach(mapper)
    
    return output_array
 
def reduce_extents(v, axis,
                   dtype_fn,
                   local_reduce_fn,
                   combine_fn):
  return ReduceExtentsExpr([v], axis, dtype_fn, local_reduce_fn, combine_fn)
