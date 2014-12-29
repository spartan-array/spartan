#!/usr/bin/env python

'''
Implementation of the ``map`` operation.

Maps encompass most of the common Numpy arithmetic operators and
element-wise operations.  For instance ``a + b`` is translated to:

::

   map((a, b), lambda x, y: x + y)

Inputs to a map are *broadcast* up to have the same shape (this is
the same behavior as Numpy broadcasting).
'''

import collections
from traits.api import Instance

from spartan import rpc
from .. import util, blob_ctx
from ..array import distarray, tile, extent
from ..core import LocalKernelResult
from ..node import indent
from ..util import Assert
from .base import ListExpr, TupleExpr, PythonValue, Expr, as_array, NotShapeable
from .broadcast import Broadcast, broadcast
from .local import (FnCallExpr, LocalInput, LocalCtx, LocalExpr, LocalMapExpr,
                    LocalMapLocationExpr, make_var)
import time


def get_local_values(ex, children, child_to_var):
  local_values = {}
  for child, childv in zip(children, child_to_var):
    if isinstance(child, Broadcast):
      # When working with a broadcasted array, it is more efficient to fetch
      #   the section of the non-broadcasted array and have NumPy broadcast
      #   internally, than to broadcast ahead of time.
      local_val = child.fetch_base_tile(ex)
    else:
      local_val = child.fetch(ex)
    local_values[childv] = local_val

  return local_values


def tile_mapper(ex, children, child_to_var, op):
  '''
  Run for each tile of a `Map` operation.

  Evaluate the map function on the local tile and return a result.

  :param ex: `Extent`
  :param children: Input arrays for this operation.
  :param child_to_var: Map from a child to the varname.
  :param op: `LocalExpr` to evaluate.
  '''
  local_values = get_local_values(ex, children, child_to_var)
  local_values['extent'] = ex

  #util.log_info('MapTiles: %s', op)
  #util.log_info('Fetching %d inputs', len(children))
  #util.log_info('%s %s', children, ex)

  #local_values = dict([(k, v.fetch(ex)) for (k, v) in children.iteritems()])
  #util.log_info('Local %s', [type(v) for v in local_values.values()])
  #util.log_info('Local %s', local_values)
  #util.log_info('Op %s', op)

  op_ctx = LocalCtx(inputs=local_values)

  #util.log_info('Inputs: %s', local_values)
  result = op.evaluate(op_ctx)

  if id(result) == id(local_values[child_to_var[0]]):
    return LocalKernelResult(result=[(ex, children[0].tiles[ex])])

  #util.log_info('Result: %s', result)
  Assert.eq(ex.shape, result.shape,
            'Bad shape -- source = %s, result = %s, op = (%s)',
            local_values, result, op)

  # make a new tile and return it
  result_tile = tile.from_data(result)
  tile_id = blob_ctx.get().create(result_tile).wait().tile_id

  return LocalKernelResult(result=[(ex, tile_id)])


class MapExpr(Expr):
  '''Represents mapping an operator over one or more inputs.

  :ivar op: A `LocalExpr` to evaluate on the input(s)
  :ivar children: One or more `Expr` to map over.
  '''
  children = Instance(ListExpr)
  child_to_var = Instance(list)
  op = Instance(LocalExpr)

  def pretty_str(self):
    return 'Map[%d](%s, %s)' % (self.expr_id, self.op.pretty_str(),
                                indent(self.children.pretty_str()))

  def compute_shape(self):
    '''MapTiles retains the shape of inputs.

    Broadcasting results in a map taking the shape of the largest input.

    Right align matrices according to numpy broadcasting rules. See `broadcast`
    in spartan/expr/broadcast.py for reference.

    '''
    orig_shapes = [list(x.shape) for x in self.children]
    dims = [len(shape) for shape in orig_shapes]
    max_dim = max(dims)
    new_shapes = []

    # prepend filler dimensions for smaller arrays
    for shp in orig_shapes:
      diff = max_dim - len(shp)
      new_shapes.append([1] * diff + shp)

    output_shape = collections.defaultdict(int)
    for s in new_shapes:
      for i, v in enumerate(s):
        output_shape[i] = max(output_shape[i], v)
    return tuple([output_shape[i] for i in range(len(output_shape))])

  def _evaluate_kw(self, op):
    '''
    Evaluate all the exprs in the map kws. It is used mostly by region_map which will contain exprs in its kws.
    It can avoid expr tree optimization being interrupted by the region_map. This evaluation will not affect the
    map fusion optimization. It just turns the exprs in the kws into DistArray.

    Args:
      op (LocalExpr): the map Local operations.
    '''
    if isinstance(op, FnCallExpr) and 'fn_kw' in op.kw:
      for k, v in op.kw['fn_kw'].iteritems():
        if isinstance(v, Expr):
          if hasattr(v, 'op'): self._evaluate_kw(v.op)
          op.kw['fn_kw'][k] = v.evaluate()

    for d in op.deps:
      if isinstance(d, FnCallExpr):
        self._evaluate_kw(d)

  def _evaluate(self, ctx, deps):
    children = deps['children']
    child_to_var = deps['child_to_var']
    self._evaluate_kw(self.op)
    util.log_debug('Evaluating %s.%d', self.op.fn_name(), self.expr_id)

    children = broadcast(children)
    largest = distarray.largest_value(children)

    i = children.index(largest)
    children[0], children[i] = children[i], children[0]
    child_to_var[0], child_to_var[i] = child_to_var[i], child_to_var[0]

    for child in children:
      util.log_debug('Map children: %s', child)

    #util.log_info('Mapping %s over %d inputs; largest = %s', op, len(children), largest.shape)

    return largest.map_to_array(tile_mapper, kw={'children': children,
                                                 'child_to_var': child_to_var,
                                                 'op': self.op})


def map(inputs, fn, numpy_expr=None, fn_kw=None):
  '''
  Evaluate ``fn`` over each tile of the input.

  :param inputs: list
    List of ``Expr``s to map over.
  :param fn: function
    Mapper function. Should have signature (NumPy array) -> NumPy array
  :param fn_kw: dict, Optional
    Keyword arguments to pass to ``fn``.

  :rtype: MapExpr
    An expression node representing mapping ``fn`` over ``inputs``.

  '''
  assert fn is not None

  if not util.is_iterable(inputs):
    inputs = [inputs]

  op_deps = []
  children = []
  child_to_var = []
  for v in inputs:
    v = as_array(v)
    varname = make_var()
    children.append(v)
    child_to_var.append(varname)
    op_deps.append(LocalInput(idx=varname))

  children = ListExpr(vals=children)
  op = LocalMapExpr(fn=fn, kw=fn_kw, pretty_fn=numpy_expr, deps=op_deps)

  return MapExpr(children=children, child_to_var=child_to_var, op=op)


def tile_reshape_mapper(ex, arrays, local_user_fn, local_user_fn_kw, target):
  tiles = []
  # Fetch the extents
  for i in range(0, len(arrays)):
    tiles.append(arrays[i].fetch(ex))

  if local_user_fn_kw is None:
    local_user_fn_kw = {}
  result = local_user_fn(ex, tiles, **local_user_fn_kw)
  futures = rpc.FutureGroup()
  if result is not None:
    for ex, v in result:
      futures.append(target.update(ex, v, wait=False))

  return LocalKernelResult(result=[], futures=futures)


class Map2Expr(Expr):
  arrays = Instance(TupleExpr)
  fn = PythonValue
  fn_kw = PythonValue
  shape = Instance(tuple)
  tile_hint = Instance(tuple)
  reducer = PythonValue

  def pretty_str(self):
    return 'Map2[%d]' % (self.expr_id)

  def compute_shape(self):
    return self.shape

  def _evaluate(self, ctx, deps):
    arrays = deps['arrays']
    fn = deps['fn']
    fn_kw = deps['fn_kw']
    shape = deps['shape']
    tile_hint = deps['tile_hint']
    reducer = deps['reducer']

    target = distarray.create(shape, arrays[0].dtype,
                              sharder=None, reducer=reducer,
                              tile_hint=tile_hint,
                              sparse=arrays[0].sparse)

    arrays[0].foreach_tile(mapper_fn=tile_reshape_mapper,
                           kw=dict(arrays=arrays, local_user_fn=fn,
                                   local_user_fn_kw=fn_kw,
                                   target=target))
    return target


def map2(arrays, fn, fn_kw=None, shape=None, tile_hint=None, reducer=None):
  '''
  Arguments:
    arrays:
    fn:
    fn_kw:
    shape:
    tile_hint:
    reducer:

  Returns:
    Map2Expr
  '''
  if not util.is_iterable(arrays):
    arrays = [arrays]

  assert fn is not None
  assert shape is not None

  arrays = tuple(arrays)
  arrays = TupleExpr(vals=arrays)
  return Map2Expr(arrays=arrays, fn=fn, fn_kw=fn_kw,
                  shape=tuple(shape), tile_hint=tile_hint,
                  reducer=reducer)


def join_mapper(ex, arrays, axes, local_user_fn, local_user_fn_kw, target):
  # First find out extents for all arrays
  first_extent = extent.change_partition_axis(ex, axes[0])
  keys = (first_extent.ul[axes[0]], first_extent.lr[axes[0]])
  join_extents = [first_extent]
  for i in range(1, len(arrays)):
    ul = [0 for j in range(len(arrays[i].shape))]
    lr = list(arrays[i].shape)
    ul[axes[i]] = keys[0]
    lr[axes[i]] = keys[1]
    join_extents.append(extent.create(ul, lr, arrays[i].shape))

  tiles = []
  # Fetch the extents
  for i in range(0, len(arrays)):
    tiles.append(arrays[i].fetch(join_extents[i]))

  if local_user_fn_kw is None:
    local_user_fn_kw = {}
  result = local_user_fn(join_extents, tiles, **local_user_fn_kw)
  futures = rpc.FutureGroup()
  if result is not None:
    for ex, v in result:
      futures.append(target.update(ex, v, wait=False))

  return LocalKernelResult(result=[], futures=futures)


class Map3Expr(Expr):
  arrays = Instance(TupleExpr)
  axes = Instance(tuple)
  fn = PythonValue
  fn_kw = PythonValue
  shape = Instance(tuple)
  tile_hint = Instance(tuple)
  reducer = PythonValue

  def pretty_str(self):
    return 'Map3[%d]' % (self.expr_id)

  def compute_shape(self):
    return self.shape

  def _evaluate(self, ctx, deps):
    arrays = deps['arrays']
    axes = deps['axes']
    fn = deps['fn']
    fn_kw = deps['fn_kw']
    shape = deps['shape']
    tile_hint = deps['tile_hint']
    reducer = deps['reducer']

    target = distarray.create(shape, arrays[0].dtype,
                              sharder=None, reducer=reducer,
                              tile_hint=tile_hint,
                              sparse=arrays[0].sparse)

    arrays[0].foreach_tile(mapper_fn=join_mapper,
                           kw=dict(arrays=arrays, axes=axes, local_user_fn=fn,
                                   local_user_fn_kw=fn_kw, target=target))
    return target


def map3(arrays, axes, fn, fn_kw=None, shape=None, tile_hint=None, reducer=None):
  '''
  Arguments:
    arrays:
    axes:
    fn:
    fn_kw:
    shape:
    tile_hint:
    reducer:

  Returns:
    Map3Expr
  '''
  if not util.is_iterable(arrays):
    arrays = [arrays]

  if not util.is_iterable(axes):
    axes = [axes]

  assert fn is not None
  assert len(arrays) == len(axes)
  assert shape is not None

  arrays = tuple(arrays)
  arrays = TupleExpr(vals=arrays)
  axes = tuple(axes)
  return Map3Expr(arrays=arrays, axes=axes, fn=fn, fn_kw=fn_kw,
                  shape=tuple(shape), tile_hint=tile_hint,
                  reducer=reducer)
