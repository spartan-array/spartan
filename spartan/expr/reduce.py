import numpy as np

from spartan import util
from spartan.array import extent, distarray
from spartan.expr import local
from spartan.expr.local import make_var, LocalReduceExpr, LocalInput, LocalCtx
from spartan.node import Node, node_type
from spartan.util import Assert

from .base import Expr, DictExpr
from .map import MapResult
from ..rpc import TimeoutException

def _reduce_mapper(ex, children, op, axis, output):
  '''Run a local reducer for a tile, and update the appropiate 
  portion of the output array.
  
  N.B. Scipy sparse matrices DO NOT support dimensions != 2.
  As a result, naive reductions over these matrices will fail
  as the local reduction value will not have the correct
  dimensionality.
  
  To deal with this, we assume that all outputs of a reduction
  will be dense: we convert sparse reduction outputs to dense
  and fix the dimensions before updating the target array.
  '''
  
  #util.log_info('Reduce: %s %s %s %s %s', reducer, ex, tile, axis, fn_kw)

  local_values = dict([(k, v.fetch(ex)) for k, v in children.iteritems()])
  
  # Set extent and axis information for user functions
  local_values['extent'] = ex
  local_values['axis'] = axis

  ctx = LocalCtx(inputs=local_values)

  local_reduction = op.evaluate(ctx)
  dst_extent = extent.index_for_reduction(ex, axis)
  
  # HACK -- scipy.sparse matrices output DENSE values
  # with the WRONG shape.  Fix shapes here that have
  # the right SIZE but wrong number of dimensions.
  # ARGH.
  #if scipy.sparse.issparse(local_reduction):
  #  local_reduction = local_reduction.todense()
  Assert.eq(local_reduction.size, dst_extent.size)
  
  # fix shape
  local_reduction = np.asarray(local_reduction).reshape(dst_extent.shape)
    
  #util.log_info('Update: %s %s', dst_extent, local_reduction)
  output.update(dst_extent, local_reduction)
  return MapResult([], None)

@node_type
class ReduceExpr(Expr):
  _members = ['children', 'axis', 'dtype_fn', 'op', 'accumulate_fn']
  
  def node_init(self):
    Expr.node_init(self)
    assert self.dtype_fn is not None
    assert isinstance(self.children, DictExpr)
  
  def _evaluate(self, ctx, deps):
    children = deps['children']
    axis = deps['axis']
    op = deps['op']
    tile_accum = deps['accumulate_fn']

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
    
    try:
      output_array = distarray.create(shape, dtype,
                                    reducer=tile_accum)

    # util.log_info('Reducing into array %s', output_array)
      largest.foreach_tile(_reduce_mapper, kw={'children' : children,
                                               'op' : op,
                                               'axis' : axis,
                                               'output' : output_array})
    except TimeoutException as ex:
      util.log_info('reduce expr %d need to retry' % self.expr_id)
      return self.evaluate()

    return output_array

 
def reduce(v, axis, dtype_fn, local_reduce_fn, accumulate_fn, fn_kw=None):
  '''
  Reduce ``v`` over axis ``axis``.
  
  The resulting array should have a datatype given by ``dtype_fn(input).``
  
  For each tile of the input ``local_reduce_fn`` is called with
  arguments: (tiledata, axis, extent).

  The output is combined using ``accumulate_fn``.
   
  :param v: `Expr`
  :param axis: int or None
  :param dtype_fn: Callable: fn(array) -> `numpy.dtype`
  :param local_reduce_fn: Callable: fn(extent, data, axis)
  :param accumulate_fn: Callable: fn(old_v, update_v) -> new_v 
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
                    accumulate_fn=accumulate_fn)

