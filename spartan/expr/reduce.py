from .base import Expr, Op, OpCtx, make_var, DictExpr, OpInput
from spartan import util
from spartan.array import extent, tile, distarray
from spartan.node import Node

class ReduceOp(Op):
  __metaclass__ = Node
  _members = ['fn', 'deps', 'kw']

  def evaluate(self, ctx):
    deps = [d.evaluate(ctx) for d in self.deps]
    assert len(deps) == 1
    if self.kw is None: self.kw = {}
    return self.fn(ctx.extent, deps[0], axis=ctx.axis, **self.kw)

def _reduce_mapper(ex, children, op, axis, output):
  #util.log_info('Reduce: %s %s %s %s %s', reducer, ex, tile, axis, fn_kw)

  local_values = dict([(k, v.fetch(ex)) for k, v in children.iteritems()])
  ctx = OpCtx(inputs=local_values,
              axis=axis,
              extent=ex)

  reduced = op.evaluate(ctx)
  dst_extent = extent.index_for_reduction(ex, axis)
  #util.log_info('Update: %s %s', dst_extent, reduced)
  output.update(dst_extent, reduced)
  return []

class ReduceExpr(Expr):
  __metaclass__ = Node
  _members = ['children', 'axis', 'dtype_fn', 'op', 'combine_fn']
  
  def evaluate(self, ctx, deps):
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
  
  For each tile of the input ``local_reduce_fn`` is called.
  The output is combined using ``combine_fn``.
   
  :param v: `Expr`
  :param axis: int or None
  :param dtype_fn: Callable: fn(array) -> `numpy.dtype`
  :param local_reduce_fn: Callable: fn(extent, data, axis)
  :param combine_fn: Callable: fn(old_v, update_v) -> new_v 
  '''
  if fn_kw is None: fn_kw = {}
  varname = make_var()

  reduce_op = ReduceOp(fn=local_reduce_fn,
                       deps=[OpInput(idx=varname)],
                       kw=fn_kw)

  return ReduceExpr(children=DictExpr({ varname : v}),
                    axis=axis,
                    dtype_fn=dtype_fn,
                    op=reduce_op,
                    combine_fn=combine_fn)
