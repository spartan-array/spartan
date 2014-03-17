#!/usr/bin/env python


'''
Optimizations over an expression graph.

Optimization passes take as input an expression graph, and return a
(hopefully) simpler, equivalent graph.  This module defines the
pass infrastructure, the fusion passes and an optimization pass to
lower code to Parakeet.
'''

from ..config import FLAGS, BoolFlag
from . import local
from .filter import FilterExpr
from .slice import SliceExpr
from .local import make_var, LocalInput, ParakeetExpr
from .reduce import ReduceExpr, LocalReduceExpr
from ..util import Assert

from .. import util
from .base import Expr, Val, AsArray, DictExpr, lazify, expr_like, ExprTrace
from .map import MapExpr, LocalMapExpr
from .ndarray import NdArrayExpr
from .shuffle import ShuffleExpr
from .write_array import WriteArrayExpr


try:
  import numexpr
except:
  numexpr = None

try:
  import parakeet
except:
  parakeet = None


_parakeet_blacklist = set()

def disable_parakeet(fn):
  "Disables parakeet optimization for this function."
  _parakeet_blacklist.add(fn)
  return fn

_not_idempotent = set()

def not_idempotent(fn):
  "Disable map fusion for ``fn``."
  _not_idempotent.add(fn)
  return fn


class OptimizePass(object):
  # These are used to ensure passes are ordered correctly.
  before = []
  after = []

  def __init__(self):
    self.visited = {}

  def visit(self, op):
    if not isinstance(op, Expr):
      return op

    if op in self.visited:
      return self.visited[op]

    #assert not op in self.visited, 'Infinite recursion during optimization %s' % op
    #self.visited.add(op)

    #util.log_info('VISIT %s: %s', op.typename(), hash(op))
    if hasattr(self, 'visit_default'):
      self.visited[op] = self.visit_default(op)
    elif hasattr(self, 'visit_%s' % op.typename()):
      self.visited[op] = getattr(self, 'visit_%s' % op.typename())(op)
    else:
      self.visited[op] = op.visit(self)

    return self.visited[op]


def fusable(v):
  return isinstance(v, (MapExpr,
                        ReduceExpr,
                        ShuffleExpr,
                        NdArrayExpr,
                        SliceExpr,
                        FilterExpr,
                        Val,
                        AsArray,
                        WriteArrayExpr))


def merge_var(children, k, v):
  """Add a new expression with key ``k`` to the ``children`` dictionary.

  If ``k`` is already in the dictionary, than ``v`` must be equal to
  the current value.
  """
  if k in children:
    Assert.eq(v, children[k])
  else:
    children[k] = v


class MapMapFusion(OptimizePass):
  '''Fold sequences of Map operations together.
  
  map(f, map(g, map(h, x))) -> map(f . g . h, x)
  '''
  name = 'map_fusion'

  def visit_MapExpr(self, expr):
    #util.log_info('Map tiles: %s', expr)
    Assert.iterable(expr.children)
    #util.log_info('VISIT %d', id(op))
    #util.log_info('VISIT %s', op.children)
    map_children = self.visit(expr.children)
    all_maps = True
    Assert.isinstance(map_children, DictExpr)
    for k, v in map_children.iteritems():
      if not fusable(v):
        print 'Skipping fusion: (%s -> %s)' % (k, type(v))
        all_maps = False
        break

    if (not all_maps
        or isinstance(expr.op, local.ParakeetExpr)
        or expr.op.fn in _not_idempotent):
      return expr.visit(self)

    #util.log_info('Original: %s', expr.op)
    children = {}
    combined_op = LocalMapExpr(fn=expr.op.fn,
                               kw=expr.op.kw,
                               pretty_fn=expr.op.pretty_fn)
    trace = ExprTrace()
    for name, child_expr in map_children.iteritems():
      trace.fuse(child_expr.stack_trace)
      if isinstance(child_expr, MapExpr):
        for k, v in child_expr.children.iteritems():
          merge_var(children, k, v)

        #util.log_info('Merging: %s', child_expr.op)
        combined_op.add_dep(child_expr.op)
      else:
        key = make_var()
        combined_op.add_dep(LocalInput(idx=key))
        children[key] = child_expr

    return expr_like(expr,
                     children=DictExpr(vals=children),
                     op=combined_op,
                     trace=trace)


class ReduceMapFusion(OptimizePass):
  '''Fuse reduce(f, map(g, X)) -> reduce(f . g, X)'''
  name = 'reduce_fusion'
  after = [MapMapFusion]

  def visit_ReduceExpr(self, expr):
    Assert.isinstance(expr.children, DictExpr)
    old_children = self.visit(expr.children)

    for k, v in old_children.iteritems():
      if not isinstance(v, MapExpr):
        return expr.visit(self)

    combined_op = LocalReduceExpr(fn=expr.op.fn,
                                  kw=expr.op.kw,
                                  deps=[expr.op.deps[0]])

    new_children = {}
    trace = ExprTrace()
    for name, child_expr in old_children.iteritems():
      for k, v in child_expr.children.iteritems():
        merge_var(new_children, k, v)
      combined_op.add_dep(child_expr.op)
      trace.fuse(child_expr.stack_trace)

    return expr_like(expr,
                     children=DictExpr(vals=new_children),
                     axis=expr.axis,
                     dtype_fn=expr.dtype_fn,
                     accumulate_fn=expr.accumulate_fn,
                     op=combined_op,
                     trace=trace)


class SliceMapFusion(OptimizePass):
  name = 'slice_map_fusion'
  after = [MapMapFusion]

  def visit_SliceExpr(self, expr):
    pass

class CollapsedCachedExpressions(OptimizePass):
  '''Replace expressions which have already been evaluated
  with a simple value expression.

  This results in simpler local expressions when evaluating
  iterative programs.
  '''

  name = 'collapse_cached'
  before = [MapMapFusion, ReduceMapFusion]

  def visit_default(self, expr):
    #util.log_info('Visit: %s, %s', expr.expr_id, expr.cache)
    cache = expr.cache()
    if cache is not None:
      util.log_info('Collapsing %s', expr.typename())
      return lazify(cache)
    else:
      return expr.visit(self)


def _find_modules(op):
  '''Find any modules referenced by the given `LocalOp` or its dependencies'''
  modules = set()
  if isinstance(op, local.FnCallExpr):
    if hasattr(op.fn, '__module__'):
      modules.add(op.fn.__module__)
    elif hasattr(op.fn, '__class__'):
      modules.add(op.fn.__class__.__module__)


  for d in op.deps:
    modules.union(_find_modules(d))

  return modules


def _parakeet_codegen(op):
  '''Given a local operation, generate an equivalent parakeet function definition.'''

  def _codegen(op):
    if isinstance(op, local.FnCallExpr):
      if util.is_lambda(op.fn) and not op.pretty_fn:
        raise local.CodegenException('Cannot codegen through a lambda expression: %s' % op.fn)

      if op.fn in _parakeet_blacklist:
        raise local.CodegenException('Blacklisted %s', op.fn)

      name = op.fn_name()

      arg_str = ','.join([_codegen(v) for v in op.deps])
      kw_str = ','.join(['%s=%s' % (k, repr(v)) for k, v in op.kw.iteritems()])
      if arg_str:
        kw_str = ',' + kw_str

      return '%s(%s %s)' % (name, arg_str, kw_str)
    elif isinstance(op, local.LocalInput):
      return op.idx
    else:
      raise local.CodegenException('Cannot codegen for %s' % type(op))

  if isinstance(op, ParakeetExpr):
    return op.source

  op_code =  _codegen(op)

  module_prelude = [
    'import parakeet',
    'import spartan.expr',
    'import numpy',
    'from spartan import mathlib',
    'from spartan import util',
  ]

  for mod in _find_modules(op):
    module_prelude.append('import %s' % mod)

  fn_prelude = '''
@util.synchronized
@parakeet.jit
'''

  fn = '\n'.join(module_prelude)
  fn = fn + fn_prelude
  fn = fn + 'def _jit_fn'
  fn = fn + '(%s):\n  ' % ','.join(op.input_names())
  fn = fn + 'return ' + op_code
  fn = fn + '\n\n'

  # verify we can compile before proceeding
  local.compile_parakeet_source(fn)
  return fn


class ParakeetGeneration(OptimizePass):
  '''
  Replace local map/reduce operations with an equivalent
  parakeet function definition.
  '''

  name = 'parakeet_gen'
  after = [MapMapFusion, ReduceMapFusion]

  def visit_MapExpr(self, expr):
    # if we've already converted this to parakeet, stop now
    if isinstance(expr.op, local.ParakeetExpr):
      return expr.visit(self)

    try:
      source = _parakeet_codegen(expr.op)
      return expr_like(expr,
                       op=local.ParakeetExpr(source=source,  deps=expr.op.deps),
                       children=expr.children)
    except local.CodegenException:
      util.log_info('Failed to convert to parakeet.')
      return expr.visit(self)

#   Parakeet doesn't support taking the current Extent object as
#   an argument, which prevents enabling it for reductions.
#
#   def visit_ReduceExpr(self, expr):
#     # if we've already converted this to parakeet, stop now
#     if isinstance(expr.op, local.ParakeetExpr):
#       return expr.visit(self)
# 
#     try:
#       source = codegen(expr.op)
#       return expr_like(expr,
#                        children=expr.children,
#                        axis=expr.axis,
#                        dtype_fn=expr.dtype_fn,
#                        accumulate_fn=expr.accumulate_fn,
#                        op=local.ParakeetExpr(source=source,  deps=expr.op.deps))
#     except local.CodegenException:
#       util.log_info('Failed to convert to parakeet.')
#       return expr.visit(self)


class RotateSlice(OptimizePass):
  '''
  This pass rotates slice operations to the bottom of the expression graph.

  By moving slices down, we open up opportunities for performing more
  fusion operations.

  The actual operation performed is:

  (a + b)[slice] -> broadcast(a, shape)[slice] + broadcast(b, shape)[slice]
  '''
  name = 'rotate_slice'

  def visit_SliceExpr(self, slice_expr):
    'Rotate this slice with a child map expression.'
    map_expr = slice_expr.src

    if not isinstance(map_expr, MapExpr):
      return slice_expr.visit(self)

    try:
      map_shape = map_expr.compute_shape()
    except base.NotShapeable:
      return slice_expr.visit(self)

    Assert.iterable(map_expr.children)
    map_children = self.visit(map_expr.children)

    children = {}
    for name, child_expr in map_children.iteritems():
      children[name] = SliceExpr(src=child_expr,
                                 idx=slice_expr.idx,
                                 broadcast_to=map_shape)

    return expr_like(map_expr,
                     op=map_expr.op,
                     children=DictExpr(vals=children),
                     trace=map_expr.stack_trace)



def apply_pass(klass, dag):
  if not getattr(FLAGS, 'opt_' + klass.name):
    util.log_debug('Pass %s disabled', klass.name)
    return dag

  p = klass()
  return p.visit(dag)


passes = []


def optimize(dag):
  if not FLAGS.optimization:
    util.log_info('Optimizations disabled')
    return dag

  for p in passes:
    dag = apply_pass(p, dag)

  return dag


def add_optimization(klass, default):
  passes.append(klass)

  for p in passes:
    assert (not klass in p.before), 'Invalid ordering of passes %s' % passes
    assert (not klass in p.after), 'Invalid ordering of passes %s' % passes

  flagname = 'opt_' + klass.name
  #setattr(Flags, flagname, add_bool_flag(flagname, default=default))
  FLAGS.add(BoolFlag(flagname, default=default, help='Enable %s optimization' % klass.__name__))

  #util.log_info('Passes: %s', passes)


add_optimization(MapMapFusion, True)
add_optimization(ReduceMapFusion, True)
add_optimization(CollapsedCachedExpressions, True)
add_optimization(RotateSlice, False)

if parakeet is not None:
  add_optimization(ParakeetGeneration, True)

FLAGS.add(BoolFlag('optimization', default=True))
