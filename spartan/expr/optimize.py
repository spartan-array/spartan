#!/usr/bin/env python

'''
Optimizations over an expression graph.
'''
import tempfile

import numpy as np
from spartan.config import FLAGS, BoolFlag
from spartan.expr import local
from spartan.expr.local import make_var, LocalInput, ParakeetExpr
from spartan.expr.reduce import ReduceExpr, LocalReduceExpr
from spartan.util import Assert

from .. import util
from .base import Expr, Val, ListExpr, AsArray, DictExpr, lazify, expr_like
from .shuffle import ShuffleExpr
from .map import MapExpr, LocalMapExpr
from .ndarray import NdArrayExpr
from spartan.array.distarray import broadcast

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


def map_like(v):
  return isinstance(v, (MapExpr, ShuffleExpr, NdArrayExpr, Val, AsArray))


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
      if not map_like(v):
        all_maps = False
        break

    if not all_maps:
      return expr.visit(self)

    #util.log_info('Original: %s', expr.op)
    children = {}
    combined_op = LocalMapExpr(fn=expr.op.fn,
                               kw=expr.op.kw,
                               pretty_fn=expr.op.pretty_fn)
    for name, child_expr in map_children.iteritems():
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
                     op=combined_op)


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
    for name, child_expr in old_children.iteritems():
      for k, v in child_expr.children.iteritems():
        merge_var(new_children, k, v)
      combined_op.add_dep(child_expr.op)

    return expr_like(expr,
                     children=new_children,
                     axis=expr.axis,
                     dtype_fn=expr.dtype_fn,
                     combine_fn=expr.combine_fn,
                     op=combined_op)


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
    if expr.cache is not None:
      util.log_info('Collapsing %s', expr.typename())
      return lazify(expr.cache)
    else:
      return expr.visit(self)

def _codegen(op):
  if isinstance(op, local.FnCallExpr):
    if util.is_lambda(op.fn) and not op.pretty_fn:
      raise local.CodegenException('Cannot codegen through a lambda expression: %s' % op.fn)

    if op.fn in _parakeet_blacklist:
      raise local.CodegenException('Blacklisted %s', op.fn)

    name = op.fn_name()

    arg_str = ','.join([_codegen(v) for v in op.deps])
    kw_str = ','.join(['%s=%s' % (k, v) for k, v in op.kw.iteritems()])
    if arg_str:
      kw = ',' + kw_str

    return '%s(%s %s)' % (name, arg_str, kw_str)
  elif isinstance(op, local.LocalInput):
    return op.idx
  else:
    raise local.CodegenException('Cannot codegen for %s' % type(op))


def codegen(op):
  '''Given a local operation, generate an equivalent parakeet function definition.'''
  if isinstance(op, ParakeetExpr):
    return op.source

  op_code =  _codegen(op)

  prelude = '\n'.join([
    'import parakeet',
    'import spartan.expr',
    'import numpy',
    'from spartan import mathlib',
    'from spartan import util',
    '@util.synchronized',
    '@parakeet.jit',
  ])

  fn = prelude + 'def _jit_fn'
  fn = fn + '(%s):\n  ' % ','.join(op.input_names())
  fn = fn + 'return ' + op_code

  # verify we can compile before proceeding
  local.compile_parakeet_source(fn)
  return fn

class ParakeetGeneration(OptimizePass):
  '''Currently disabled.

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
      source = codegen(expr.op)
      return expr_like(expr,
                       op=local.ParakeetExpr(source=source,  deps=expr.op.deps),
                       children=expr.children)
    except local.CodegenException:
      util.log_info('Failed to convert to parakeet.', exc_info=1)
      return expr.visit(self)


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

if parakeet is not None:
  add_optimization(ParakeetGeneration, True)

FLAGS.add(BoolFlag('optimization', default=True))
