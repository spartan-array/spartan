from .base import Expr, DictExpr
from spartan.array import extent, distarray
from spartan.expr import local
from spartan.expr.local import make_var, LocalReduceExpr, LocalInput, LocalCtx
from spartan.node import Node

def _reduce_mapper(ex, children, op, axis, output):
  #util.log_info('Reduce: %s %s %s %s %s', reducer, ex, tile, axis, fn_kw)

  local_values = dict([(k, v.fetch(ex)) for k, v in children.iteritems()])
  local_values['extent'] = ex
  local_values['axis'] = axis

  ctx = LocalCtx(inputs=local_values)

  reduced = op.evaluate(ctx)
  dst_extent = extent.index_for_reduction(ex, axis)
  #util.log_info('Update: %s %s', dst_extent, reduced)
  output.update(dst_extent, reduced)
  return []

class ReduceExpr(Expr):
  __metaclass__ = Node
  _members = ['children', 'axis', 'dtype_fn', 'op', 'combine_fn']
  
  def _evaluate(self, ctx, deps):
    children = deps['children']
    axis = deps['axis']
    op = deps['op']
    tile_accum = deps['combine_fn']

    keys = children.keys()
    vals = children.values()
    vals = distarray.broadcast(vals)
    largest = distarray.largest_value(vals)
    children = dict(zip(keys, vals))

    dtype = deps['dtype_fn'](vals[0])
    # util.log_info('Reducer: %s', op)
    # util.log_info('Combiner: %s', tile_accum)
    # util.log_info('Reducing %s over axis %s', children, axis)

    shape = extent.shape_for_reduction(vals[0].shape, axis)
    output_array = distarray.create(shape, dtype,
                                    combiner=tile_accum, 
                                    reducer=tile_accum)

    #print local.codegen(op)
    
    # util.log_info('Reducing into array %s', output_array)
    largest.foreach(_reduce_mapper, kw={
      'children' : children,
      'op' : op,
      'axis' : axis,
      'output' : output_array})

    return output_array

 
def reduce(v, axis, dtype_fn, local_reduce_fn, combine_fn, fn_kw=None):
  '''
  Reduce ``v`` over axis ``axis``.
  
  The resulting array should have a datatype given by ``dtype_fn(input).``
  
  For each tile of the input ``local_reduce_fn`` is called with
  arguments: (tiledata, axis, extent).

  The output is combined using ``combine_fn``.
   
  :param v: `Expr`
  :param axis: int or None
  :param dtype_fn: Callable: fn(array) -> `numpy.dtype`
  :param local_reduce_fn: Callable: fn(extent, data, axis)
  :param combine_fn: Callable: fn(old_v, update_v) -> new_v 
  '''
  if fn_kw is None: fn_kw = {}
  varname = make_var()

  assert not 'axis' in fn_kw, '"axis" argument is reserved.'
  fn_kw['axis'] = axis

  reduce_op = LocalReduceExpr(fn=local_reduce_fn,
                              deps=[
                                    LocalInput(idx='extent'),
                                    LocalInput(idx=varname),
                              ],
                              kw=fn_kw)

  return ReduceExpr(children=DictExpr(vals={ varname : v}),
                    axis=axis,
                    dtype_fn=dtype_fn,
                    op=reduce_op,
                    combine_fn=combine_fn)

