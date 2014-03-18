'''Implementation of the reduction expression.

This supports generic reduce operations such as 
``sum``, ``argmin``, ``argmax``, ``min`` and ``max``.

'''
import numpy as np
import collections

from ..array import extent, distarray
from ..expr.local import make_var, LocalExpr, LocalReduceExpr, LocalInput, LocalCtx
from ..util import Assert
from . import broadcast
from .base import Expr, DictExpr
from ..core import LocalKernelResult
from traits.api import Instance, Function, PythonValue

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
  return LocalKernelResult(result=[])

class ReduceExpr(Expr):
  #_members = ['children', 'axis', 'dtype_fn', 'op', 'accumulate_fn']
  children = Instance(DictExpr) 
  axis = PythonValue(None, desc="Integer or None")
  dtype_fn = Function
  op = Instance(LocalExpr) 
  accumulate_fn = PythonValue(None, desc="Function or ReduceExpr")

  def __init__(self, *args, **kw):
    super(ReduceExpr, self).__init__(*args, **kw)
    assert self.dtype_fn is not None
    assert isinstance(self.children, DictExpr)

  def compute_shape(self):
    shapes = [i.shape for i in self.children.values()]
    child_shape = collections.defaultdict(int)
    for s in shapes:
      for i, v in enumerate(s):
        child_shape[i] = max(child_shape[i], v)
    input_shape = tuple([child_shape[i] for i in range(len(child_shape))])
    return extent.shape_for_reduction(input_shape, self.axis)
  
  def label(self):
    return 'reduce(%s)' % self.op.fn.__name__
  
  def _evaluate(self, ctx, deps):
    children = deps['children']
    axis = deps['axis']
    op = deps['op']
    tile_accum = deps['accumulate_fn']

    keys = children.keys()
    vals = children.values()
    vals = broadcast.broadcast(vals)
    largest = distarray.largest_value(vals)
    children = dict(zip(keys, vals))

    dtype = deps['dtype_fn'](vals[0])
    # util.log_info('Reducer: %s', op)
    # util.log_info('Combiner: %s', tile_accum)
    # util.log_info('Reducing %s over axis %s', children, axis)

    shape = extent.shape_for_reduction(vals[0].shape, axis)
    
    output_array = distarray.create(shape, dtype,
                                    reducer=tile_accum)

  # util.log_info('Reducing into array %s', output_array)
    largest.foreach_tile(_reduce_mapper, kw={'children' : children,
                                             'op' : op,
                                             'axis' : axis,
                                             'output' : output_array})

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
  
  :rtype: `Expr`
  
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

