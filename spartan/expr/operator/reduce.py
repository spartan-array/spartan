'''Implementation of the reduction expression.

This supports generic reduce operations such as
``sum``, ``argmin``, ``argmax``, ``min`` and ``max``.

'''
import collections
import numpy as np

from traits.api import Instance, Function, PythonValue

from spartan.node import indent
from . import broadcast
from .base import Expr, ListExpr
from .local import make_var, LocalExpr, LocalReduceExpr, LocalInput, LocalCtx
from ...array import extent, distarray
from ...util import Assert
from ...core import LocalKernelResult


def _reduce_mapper(ex, children, child_to_var, op, axis, output):
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

  local_values = {}
  for i in range(len(children)):
    if isinstance(children[i], broadcast.Broadcast):
      # When working with a broadcasted array, it is more efficient to fetch the corresponding
      # section of the non-broadcasted array and have Numpy broadcast internally, than
      # to broadcast ahead of time.
      lv = children[i].fetch_base_tile(ex)
    else:
      lv = children[i].fetch(ex)
    local_values[child_to_var[i]] = lv

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
  children = Instance(ListExpr)
  child_to_var = Instance(list)
  axis = PythonValue(None, desc="Integer or None")
  dtype_fn = Function
  op = Instance(LocalExpr)
  accumulate_fn = PythonValue(None, desc="Function or ReduceExpr")
  tile_hint = PythonValue(None, desc="Tuple or None")

  def __init__(self, *args, **kw):
    super(ReduceExpr, self).__init__(*args, **kw)
    assert self.dtype_fn is not None
    assert isinstance(self.children, ListExpr)

  def compute_shape(self):
    shapes = [i.shape for i in self.children]
    child_shape = collections.defaultdict(int)
    for s in shapes:
      for i, v in enumerate(s):
        child_shape[i] = max(child_shape[i], v)
    input_shape = tuple([child_shape[i] for i in range(len(child_shape))])
    return extent.shape_for_reduction(input_shape, self.axis)

  def pretty_str(self):
    return 'Reduce(%s, axis=%s, %s, hint=%s)' % (self.op.fn.__name__, self.axis,
                                                 indent(self.children.pretty_str()), self.tile_hint)

  def _evaluate(self, ctx, deps):
    children = deps['children']
    child_to_var = deps['child_to_var']
    axis = deps['axis']
    op = deps['op']
    tile_accum = deps['accumulate_fn']

    children = broadcast.broadcast(children)
    largest = distarray.largest_value(children)

    dtype = deps['dtype_fn'](children[0])
    # util.log_info('Reducer: %s', op)
    # util.log_info('Combiner: %s', tile_accum)
    # util.log_info('Reducing %s over axis %s', children, axis)

    shape = extent.shape_for_reduction(children[0].shape, axis)

    output_array = distarray.create(shape, dtype,
                                    reducer=tile_accum, tile_hint=self.tile_hint)

  # util.log_info('Reducing into array %s', output_array)
    largest.foreach_tile(_reduce_mapper, kw={'children': children,
                                             'child_to_var': child_to_var,
                                             'op': op,
                                             'axis': axis,
                                             'output': output_array})

    return output_array


def reduce(v, axis, dtype_fn, local_reduce_fn, accumulate_fn, fn_kw=None, tile_hint=None):
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

  assert 'axis' not in fn_kw, '"axis" argument is reserved.'
  fn_kw['axis'] = axis

  reduce_op = LocalReduceExpr(fn=local_reduce_fn,
                              deps=[LocalInput(idx='extent'),
                                    LocalInput(idx=varname), ],
                              kw=fn_kw)

  return ReduceExpr(children=ListExpr(vals=[v]),
                    child_to_var=[varname],
                    axis=axis,
                    dtype_fn=dtype_fn,
                    op=reduce_op,
                    accumulate_fn=accumulate_fn,
                    tile_hint=tile_hint)
