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

from .. import util, blob_ctx
from ..array import distarray, tile
from ..util import Assert
from .base import DictExpr, Expr, as_array
from .local import LocalExpr, LocalCtx, make_var, LocalInput, LocalMapExpr
from ..core import LocalKernelResult
from traits.api import Instance, Dict, HasTraits, PythonValue
from . import broadcast

def tile_mapper(ex, children, op):
  '''
  Run for each tile of a `Map` operation.
  
  Evaluate the map function on the local tile and return a result.
  
  :param ex: `Extent`
  :param children: Input arrays for this operation.
  :param op: `LocalExpr` to evaluate.
  '''
  ctx = blob_ctx.get()
  #util.log_info('MapTiles: %s', op)
  #util.log_info('Fetching %d inputs', len(children))
  #util.log_info('%s %s', children, ex)

  local_values = {}
  for k, gv in children.iteritems():
    lv = gv.fetch(ex)
    local_values[k] = lv

  #local_values = dict([(k, v.fetch(ex)) for (k, v) in children.iteritems()])
  #util.log_info('Local %s', [type(v) for v in local_values.values()])
  #util.log_info('Local %s', local_values)
  #util.log_info('Op %s', op)

  op_ctx = LocalCtx(inputs=local_values)

  #util.log_info('Inputs: %s', local_values)
  result = op.evaluate(op_ctx)
  
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
  children = Instance(DictExpr) 
  op = Instance(LocalExpr) 

  def label(self):
    return 'map(%s)' % self.op.fn.__name__

  def compute_shape(self):
    '''MapTiles retains the shape of inputs.

    Broadcasting results in a map taking the shape of the largest input.
    '''
    shapes = [i.shape for i in self.children.values()]
    output_shape = collections.defaultdict(int)
    for s in shapes:
      for i, v in enumerate(s):
        output_shape[i] = max(output_shape[i], v)
    return tuple([output_shape[i] for i in range(len(output_shape))])

  def _evaluate(self, ctx, deps):
    children = deps['children']
    op = self.op

    #util.log_info('Codegen for expression: %s', local.codegen(op))

    keys = children.keys()
    vals = children.values()
    vals = broadcast.broadcast(vals)
    largest = distarray.largest_value(vals)

    children = dict(zip(keys, vals))
    for k, child in children.iteritems():
      util.log_debug('Map children: %s', child)

    #util.log_info('Mapping %s over %d inputs; largest = %s', op, len(children), largest.shape)

    return largest.map_to_array(
              tile_mapper, 
              kw = { 'children' : children, 'op' : op })

def map(inputs, fn, numpy_expr=None, fn_kw=None):
  '''
  Evaluate ``fn`` over each tile of the input.
  
  Args:
    inputs (list): List of `Expr`'s to map over
    fn (function): Mapper function.  Should take a Numpy array as an input and return a new Numpy array.
    fn_kw (dict): Optional.  Keyword arguments to pass to ``fn``.
    
  Returns:
    MapExpr: An expression node representing mapping ``fn`` over ``inputs``.
  '''
  assert fn is not None
  
  if not util.is_iterable(inputs):
    inputs = [inputs]

  op_deps = []
  children = {}
  for v in inputs:
    v = as_array(v)
    varname = make_var()
    children[varname] = v
    op_deps.append(LocalInput(idx=varname))

  children = DictExpr(vals=children)
  op = LocalMapExpr(fn=fn,
                    kw=fn_kw,
                    pretty_fn=numpy_expr,
                    deps=op_deps)

  return MapExpr(children=children, op=op)


