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
from scipy import sparse as sp
from traits.api import Instance

from .. import util, blob_ctx
from ..array import distarray, tile
from ..core import LocalKernelResult
from ..node import indent
from ..util import Assert
from .base import ListExpr, Expr, as_array
from .broadcast import Broadcast, broadcast
from .local import (LocalInput, LocalCtx, LocalExpr, LocalMapExpr,
    LocalMapLocationExpr, make_var)


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

  # Not all Numpy operations are compatible with mixed sparse and dense arrays.
  #   To address this, if only one of the inputs is sparse, we convert it to
  #   dense before computing our result.
  vals = local_values.values()
  if len(vals) == 2 and sp.issparse(vals[0]) ^ sp.issparse(vals[1]):
    for (k,v) in local_values.iteritems():
      if sp.issparse(v):
        local_values[k] = v.todense()
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

  #util.log_warn('Result: %s', result)
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

  def _evaluate(self, ctx, deps):
    children = deps['children']
    child_to_var = deps['child_to_var']
    op = self.op
    util.log_debug('Evaluating %s.%d', self.op.fn_name(), self.expr_id)

    children = broadcast(children)
    largest = distarray.largest_value(children)
    
    i = children.index(largest)
    children[0], children[i] = children[i], children[0]
    child_to_var[0], child_to_var[i] = child_to_var[i], child_to_var[0]

    for child in children:
      util.log_debug('Map children: %s', child)

    #util.log_info('Mapping %s over %d inputs; largest = %s', op, len(children), largest.shape)

    return largest.map_to_array(
              tile_mapper, 
              kw = {'children':children, 'child_to_var':child_to_var, 'op':op})


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
