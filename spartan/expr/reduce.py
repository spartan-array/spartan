from .base import Expr
from spartan import util
from spartan.dense import extent, tile, distarray
from spartan.expr.base import make_primitive

def _reduce_mapper(ex, tile, reducer, axis, output, fn_kw):
  #util.log_info('Reduce: %s %s %s %s %s', reducer, ex, tile, axis, fn_kw)
  reduced = reducer(ex, tile, axis, **fn_kw)
  dst_extent = extent.index_for_reduction(ex, axis)
  #util.log_info('Update: %s %s', dst_extent, reduced)
  output.update(dst_extent, reduced)

class ReduceExpr(Expr):
  _members = ['array', 'axis', 'dtype_fn', 'reduce_fn', 'combine_fn', 'fn_kw']
  
  def evaluate(self, ctx, deps):
    input_array = deps['array']
    dtype = deps['dtype_fn'](input_array)
    axis = deps['axis']
    reducer = deps['reduce_fn']
    fn_kw = deps['fn_kw']
    
    util.log_info('Reducing %s over axis %s', input_array.shape, axis)
    
    shape = extent.shape_for_reduction(input_array.shape, axis)
    tile_accum = deps['combine_fn']
    
    output_array = distarray.create(shape, dtype,
                                    combiner=tile_accum, 
                                    reducer=tile_accum)
    
    
    util.log_info('Reducing into array %s', output_array)
    input_array.foreach(_reduce_mapper, kw={'reducer' : reducer,
                                            'axis' : axis,
                                            'output' : output_array,
                                            'fn_kw' : fn_kw})
    
    return output_array

 
def reduce(v, axis, dtype_fn, local_reduce_fn, combine_fn, fn_kw=None):
  '''
  Reduce ``v`` over axis ``axis``.
  
  The resulting array should have a datatype given by ``dtype_fn(input).``
  
  For each tile of the input ``local_reduce_fn`` is called.
  The output is combined using ``combine_fn``.
   
  :param v: `Expr`
  :param axis: int or None
  :param dtype_fn: Callable: fn(array) -> `numpy.dtype`
  :param local_reduce_fn: Callable: fn(extent, data, axis)
  :param combine_fn: Callable: fn(old_v, update_v) -> new_v 
  '''

  if fn_kw is None: fn_kw = {}
  return ReduceExpr(array=v, axis=axis, 
                     dtype_fn=dtype_fn, 
                     reduce_fn=local_reduce_fn, 
                     combine_fn=combine_fn,
                     fn_kw=fn_kw)
