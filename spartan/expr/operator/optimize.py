#!/usr/bin/env python

'''
Optimizations over an expression graph.

Optimization passes take as input an expression graph, and return a
(hopefully) simpler, equivalent graph.  This module defines the
pass infrastructure, the fusion passes and an optimization pass to
lower code to Parakeet.
'''
import operator
import math
import weakref

from collections import namedtuple

from . import local
from . import tiling
from .base import Expr, Val, AsArray, ListExpr, lazify, expr_like, ExprTrace
from .base import NotShapeable, CollectionExpr
from .filter import FilterExpr
from .local import LocalInput, LocalMapExpr, LocalMapLocationExpr, make_var
from .local import ParakeetExpr
from .map import MapExpr, Map2Expr
from .ndarray import NdArrayExpr
from .outer import OuterProductExpr
from .shuffle import ShuffleExpr
from .slice import SliceExpr
from .write_array import WriteArrayExpr
from .reduce import ReduceExpr, LocalReduceExpr
from ..dot import DotExpr

from ... import util
from ...util import Assert
from ...config import FLAGS, BoolFlag
from ...array.distarray import DistArray, LocalWrapper

try:
  import numexpr
except:
  numexpr = None

try:
  import parakeet
except:
  parakeet = None


_parakeet_blacklist = set()
_tiled_exprlist = {}
_not_idempotent_list = set()


def disable_parakeet(fn):
  "Disables parakeet optimization for this function."
  _parakeet_blacklist.add(fn)
  return fn


def not_idempotent(fn):
  "Force the result of not_idempotent ``fn`` to be evaluated at once."
  def wrapped(*args, **kw):
    result = fn(*args, **kw)
    if isinstance(result, Expr):
      result.needs_cache = True
      _not_idempotent_list.add(id(result))
    return result
  return wrapped

visited_expr = {'map_fusion': weakref.WeakValueDictionary(),
                'reduce_fusion': weakref.WeakValueDictionary(),
                'collapse_cached': weakref.WeakValueDictionary(),
                'parakeet_gen': weakref.WeakValueDictionary(),
                'rotate_slice': weakref.WeakValueDictionary(),
                'auto_tiling': weakref.WeakValueDictionary()
                }


class OptimizePass(object):
  def __init__(self):
    self.visited = visited_expr[self.name]
    #self.visited = {}

  def visit(self, op):
    if not isinstance(op, Expr):
      return op

    if op.expr_id in self.visited:
      return self.visited[op.expr_id]

    #assert not op in self.visited, 'Infinite recursion during optimization %s' % op
    #self.visited.add(op)

    util.log_debug('VISIT %s: %s', op.typename(), hash(op))
    opt_op = None
    if hasattr(self, 'visit_default'):
      opt_op = self.visit_default(op)
    elif hasattr(self, 'visit_%s' % op.typename()):
      opt_op = getattr(self, 'visit_%s' % op.typename())(op)
    else:
      opt_op = op.visit(self)

    self.visited[opt_op.expr_id] = opt_op
    return opt_op


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


def merge_var(children, child_to_var, k, v):
  """Add a new expression with key ``k`` to the ``children`` dictionary.

  If ``k`` is already in the dictionary, than ``v`` must be equal to
  the current value stored there.
  """
  try:
    i = child_to_var.index(k)
    Assert.eq(v, children[i])
  except ValueError:
    children.append(v)
    child_to_var.append(k)


class MapMapFusion(OptimizePass):
  '''Fold sequences of Map operations together.

  map(f, map(g, map(h, x))) -> map(f . g . h, x)
  '''
  name = 'map_fusion'

  def visit_MapExpr(self, expr):
    Assert.iterable(expr.children)
    #util.log_info('VISIT %d', id(op))
    #util.log_info('VISIT %s', op.children)
    map_children = self.visit(expr.children)
    all_maps = True
    Assert.isinstance(map_children, ListExpr)
    for k, v in zip(expr.child_to_var, map_children):
      if not fusable(v):
        util.log_debug('Skipping fusion: (%s -> %s)' % (k, type(v)))
        all_maps = False
        break

    if not all_maps or isinstance(expr.op, local.ParakeetExpr) or \
       id(expr) in _not_idempotent_list:
      return expr.visit(self)

    #util.log_info('Original: %s', expr.op)
    children = []
    child_to_var = []
    combined_op = expr.op.__class__(fn=expr.op.fn, kw=expr.op.kw,
                                    pretty_fn=expr.op.pretty_fn)

    trace = ExprTrace()

    for child_expr in map_children:
      trace.fuse(child_expr.stack_trace)

      if isinstance(child_expr, MapExpr):
        for k, v in zip(child_expr.child_to_var, child_expr.children):
          merge_var(children, child_to_var, k, v)

        combined_op.add_dep(child_expr.op)
        util.log_debug('Fusion: %s <- %s', expr.expr_id, child_expr.expr_id)
      else:
        children.append(child_expr)
        key = make_var()
        combined_op.add_dep(LocalInput(idx=key))
        child_to_var.append(key)

    if isinstance(combined_op, LocalMapLocationExpr):
      combined_op.add_dep(LocalInput(idx='extent'))

    return expr_like(expr,
                     children=ListExpr(vals=children),
                     child_to_var=child_to_var,
                     op=combined_op,
                     trace=trace)


class ReduceMapFusion(OptimizePass):
  '''Fuse reduce(f, map(g, X)) -> reduce(f . g, X)'''
  name = 'reduce_fusion'

  def visit_ReduceExpr(self, expr):
    Assert.isinstance(expr.children, ListExpr)
    old_children = self.visit(expr.children)

    for v in old_children:
      if not isinstance(v, (MapExpr, ParakeetExpr)) or id(v) in _not_idempotent_list:
        return expr.visit(self)

    combined_op = LocalReduceExpr(fn=expr.op.fn,
                                  kw=expr.op.kw,
                                  deps=[expr.op.deps[0]])

    new_children = []
    new_child_to_var = []
    trace = ExprTrace()
    for i in range(len(old_children)):
      name = expr.child_to_var[i]
      child_expr = old_children[i]
      for j in range(len(child_expr.children)):
        k = child_expr.child_to_var[j]
        v = child_expr.children[j]
        merge_var(new_children, new_child_to_var, k, v)
      combined_op.add_dep(child_expr.op)
      trace.fuse(child_expr.stack_trace)

    return expr_like(expr,
                     children=ListExpr(vals=new_children),
                     child_to_var=new_child_to_var,
                     axis=expr.axis,
                     dtype_fn=expr.dtype_fn,
                     accumulate_fn=expr.accumulate_fn,
                     op=combined_op,
                     tile_hint=expr.tile_hint,
                     trace=trace)


class CollapsedCachedExpressions(OptimizePass):
  '''Replace expressions which have already been evaluated
  with a simple value expression.

  This results in simpler local expressions when evaluating
  iterative programs.
  '''

  name = 'collapse_cached'

  def visit_default(self, expr):
    #util.log_info('Visit: %s, %s', expr.expr_id, expr.cache)
    cache = expr.cache()
    if cache is not None:
      util.log_info('Collapsing %s %s', expr.expr_id, expr.typename())
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

  op_code = _codegen(op)

  module_prelude = ['import parakeet',
                    'import spartan.expr',
                    'import numpy',
                    'from spartan.expr import mathlib',
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

  def _should_run_parakeet(self, expr):
    has_fncallexpr = False
    for dep in expr.op.deps:
      if isinstance(dep, local.FnCallExpr):
        has_fncallexpr = True
        break

    # If all deps are not FnCallExprs and len(deps) is a small number(<=2 for now)
    # it is not worth to do code generation.
    if not has_fncallexpr and len(expr.op.deps) <= 2:
      return False
    else:
      return True

  def visit_MapExpr(self, expr):
    # if we've already converted this to parakeet, stop now
    if isinstance(expr.op, local.ParakeetExpr):
      return expr.visit(self)

    if not self._should_run_parakeet(expr):
      return expr.visit(self)

    try:
      source = _parakeet_codegen(expr.op)

      # Recursively check if any children nodes can be optimized with Parakeet
      new_children = []
      for child in expr.children:
        new_children.append(self.visit(child))

      parakeet_expr = expr_like(expr,
                                op=local.ParakeetExpr(source=source, deps=expr.op.deps),
                                children=ListExpr(vals=new_children),
                                child_to_var=expr.child_to_var)

      if id(expr) in _not_idempotent_list: _not_idempotent_list.add(id(parakeet_expr))

      return parakeet_expr
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
  rotated = {}

  def visit_SliceExpr(self, slice_expr):
    'Rotate this slice with a child map expression.'
    map_expr = slice_expr.src

    if not isinstance(map_expr, MapExpr):
      return slice_expr.visit(self)

    try:
      map_shape = map_expr.compute_shape()
    except NotShapeable:
      return slice_expr.visit(self)

    Assert.iterable(map_expr.children)
    map_children = self.visit(map_expr.children)

    children = []
    for child_expr in map_children:
      child = SliceExpr(src=child_expr,
                        idx=slice_expr.idx,
                        broadcast_to=map_shape)
      if isinstance(child_expr, MapExpr):
        # If the child is a Map, RotateSlice should traverse it again. Some
        # nodes may be traversed several times. For current implementation,
        # traversing a node several times dones't cause any problem for
        # RotationSlice.
        if self.visited.get(child_expr, None) is not None:
          del self.visited[child_expr]
        child = self.visit(child)

      children.append(child)

    if self.rotated.get(map_expr, None):
      # If this Map has been rotated, we should create a new MapExpr.
      # An example is :
      #    c = a + b
      #    d = c[1:50]
      #    e = c[2:51]
      # In this example, RotateSlice first rotate c[2:51] to a and b.
      # When RotateSlice rotates c[1:50] to a and b, it should create
      # a new Map expr(+) for a[1:50] and b[1:50]
      return MapExpr(children=ListExpr(vals=children),
                     op=map_expr.op,
                     child_to_var=map_expr.child_to_var)
    else:
      self.rotated[map_expr] = True
      return expr_like(map_expr,
                       op=map_expr.op,
                       children=ListExpr(vals=children),
                       child_to_var=map_expr.child_to_var,
                       trace=map_expr.stack_trace)


class AutomaticTiling(OptimizePass):
  '''
  Automatically partition all the arrays.
  We build a min cost max flow DAG for the expr graph, where the cost is the communication cost
  through network. We estimate the cost according to the behavior of each expr. For _builtin_ expr
  and simple map and reduce expr, we can easily estimate the cost. However, for the user defined shuffle
  expr, we can only guess the cost or let users define the cost for us.

  All Exprs:
    [Val, AsArray, DistArray]: Already partitioned array
    [NdArrayExpr]: new array needs to be partitioned
    CollectionExpr([DictExpr, ListExpr, TupleExpr]): self.vals

    [MapExpr]: self.children
    [ReduceExpr]: self.children
    [ShuffleExpr]: self.array, self.fn_kw
    [DotExpr]: self.matrix_a, self.matrix_b

    [SliceExpr, FilterExpr, CheckpointExpr, TileOpExpr]: self.src or self.array
    [WriteArrayExpr]: self.array, self.data

    [TransposeExpr, ReshapeExpr]: self.array
  '''

  name = 'auto_tiling'
  node_type = namedtuple('node_type', ['expr', 'tiling', 'children', 'parents'])
  num_node_per_group = 4
  inited = False
  cost_model = {'map': {(0, 0): 0, (0, 1): 1, (0, 2): 1,
                        (0, 3): 1, (0, 4): 1, (0, 5): 2,
                        (1, 0): 1, (1, 1): 0, (1, 2): 1,
                        (1, 3): 1, (1, 4): 2, (1, 5): 1,
                        (2, 0): 1, (2, 1): 1, (2, 2): 0,
                        (2, 3): 2, (2, 4): 1, (2, 5): 1,
                        (3, 0): 0, (3, 1): 0, (3, 2): 1,
                        (3, 3): 0, (3, 4): 1, (3, 5): 1,
                        (4, 0): 0, (4, 1): 1, (4, 2): 0,
                        (4, 3): 1, (4, 4): 0, (4, 5): 1,
                        (5, 0): 1, (5, 1): 0, (5, 2): 0,
                        (5, 3): 1, (5, 4): 1, (5, 5): 0},
                'map2': {(0, 0): 0, (0, 1): 1, (0, 2): 1, (0, -1): 1,
                         (1, 0): 1, (1, 1): 0, (1, 2): 1, (1, -1): 1,
                         (2, 0): 1, (2, 1): 1, (2, 2): 0, (2, -1): 1,
                         (3, 0): 0, (3, 1): 0, (3, 2): 1, (3, -1): 1,
                         (4, 0): 0, (4, 1): 1, (4, 2): 0, (4, -1): 1,
                         (5, 0): 1, (5, 1): 0, (5, 2): 0, (5, -1): 1}
                }

  def init(self, expr):
    self.cur_node_id = 1
    self.edges = {}
    self.nodes = {0: self.node_type([], -1, [], [])}
    self.expr_to_nodes = {}
    self.split_nodes = {}
    self.groups = []
    self.init_expr = id(expr)
    self.inited = True

    self.tiled_exprlist = _tiled_exprlist
    #self.tiled_exprlist = {}

  def add_edge(self, edge_from, edge_to, edge_cost=0):
    #util.log_warn('add_edge:%d %d cost:%d', edge_from, edge_to, edge_cost)
    if (edge_from, edge_to) not in self.edges:
      self.nodes[edge_from].parents.append(edge_to)
      self.nodes[edge_to].children.append(edge_from)
    self.edges[(edge_from, edge_to)] = edge_cost

  def remove_edge(self, edge_from, edge_to):
    del self.edges[(edge_from, edge_to)]
    self.nodes[edge_from].parents.remove(edge_to)
    self.nodes[edge_to].children.remove(edge_from)

  def add_split_nodes(self, nodes):
    group_id = len(self.groups)
    for node in nodes:
      self.split_nodes[node] = group_id
    self.groups.append(tuple(nodes))
    #util.log_warn('add_split_nodes:%s', nodes)

  def visit_children(self, children, except_child=None):
    child_ids = []
    for child in children:
      if isinstance(child, (Expr, DistArray)) and id(child) != id(except_child):
        child_ids.extend(self.visit_default(child))
    return child_ids

  def visit_NdArrayExpr(self, expr):
    # new array need to be partitioned
    if len(expr.shape) > 1 and expr.shape[1] > 1:
      new_nodes = []

      self.nodes[self.cur_node_id] = self.node_type([expr], 0, [], [])
      new_nodes.append(self.cur_node_id)
      self.add_edge(0, self.cur_node_id, 0)
      self.cur_node_id += 1

      self.nodes[self.cur_node_id] = self.node_type([expr], 1, [], [])
      new_nodes.append(self.cur_node_id)
      self.add_edge(0, self.cur_node_id, 0)
      self.cur_node_id += 1

      self.nodes[self.cur_node_id] = self.node_type([expr], 2, [], [])
      new_nodes.append(self.cur_node_id)
      self.add_edge(0, self.cur_node_id, 0)
      self.cur_node_id += 1

      cost = reduce(operator.mul, expr.shape, 1)
      for i in range(3, self.num_node_per_group):
        self.nodes[self.cur_node_id] = self.node_type([expr], i, [], [])
        new_nodes.append(self.cur_node_id)
        self.add_edge(0, self.cur_node_id, cost)
        self.cur_node_id += 1

      self.add_split_nodes(new_nodes)
      return new_nodes
    else:
      self.nodes[self.cur_node_id] = self.node_type([expr], 0, [], [])
      self.add_edge(0, self.cur_node_id, 0)
      self.cur_node_id += 1
      return [self.cur_node_id - 1]

  def visit_MapExpr(self, expr):
    largest = max(expr.children.vals, key=lambda v: reduce(operator.mul, v.shape, 1))

    child_ids = self.visit_children([largest])
    other_child_ids = self.visit_children(expr.children.vals, largest)
    kw_ids = self.visit_children(expr.op.kw['fn_kw'].itervalues()) if 'fn_kw' in expr.op.kw else []

    # one input map, reuse child expr
    if len(other_child_ids) == 0:
      for child_id in child_ids: self.nodes[child_id].expr.append(expr)
      return child_ids

    if child_ids[0] in self.split_nodes:
      tiling_types = range(self.num_node_per_group)
      self.add_split_nodes(range(self.cur_node_id, self.cur_node_id + self.num_node_per_group))
      expr_node_ids = range(self.cur_node_id, self.cur_node_id + self.num_node_per_group)
    else:
      tiling_types = (self.nodes[child_ids[0]].tiling,)
      expr_node_ids = [self.cur_node_id]

    for (tiling_type, map_child_id) in zip(tiling_types, child_ids):
      self.nodes[self.cur_node_id] = self.node_type([expr], tiling_type, [], [])
      self.add_edge(map_child_id, self.cur_node_id, 0)

      for child_id in other_child_ids:
        child = self.nodes[child_id]
        e_cost = self.cost_model['map'][(child.tiling, tiling_type)] * reduce(operator.mul, child.expr[0].shape, 1)
        self.add_edge(child_id, self.cur_node_id, e_cost)

      for child_id in kw_ids:
        self.add_edge(child_id, self.cur_node_id, reduce(operator.mul, self.nodes[child_id].expr[0].shape, 1))
      self.cur_node_id += 1

    return expr_node_ids

  def visit_ReduceExpr(self, expr):
    child_ids = self.visit_children(expr.children.vals)
    cost = reduce(operator.mul, expr.shape, 1)
    self.nodes[self.cur_node_id] = self.node_type([expr], 0, [], [])
    for child_id in child_ids:
      child = self.nodes[child_id]
      e_cost = 0 if expr.axis is None or child.tiling == 3 or (1-expr.axis) == (child.tiling % 4) else cost
      self.add_edge(child_id, self.cur_node_id, e_cost)
    self.cur_node_id += 1
    return [self.cur_node_id - 1]

  def visit_Map2Expr(self, expr):
    child_id_groups = []
    for array in expr.arrays:
      child_id_groups.append(self.visit_children([array]))

    copy_nodes = []
    for axis, child_ids in zip(expr.axes, child_id_groups):
      if isinstance(axis, tuple): axis = 2
      if axis is None: axis = -1
      self.nodes[self.cur_node_id] = self.node_type([expr], axis, [], [])

      cost = reduce(operator.mul, self.nodes[child_ids[0]].expr[0].shape, 1)
      for child_id in child_ids:
        child = self.nodes[child_id]
        e_cost = self.cost_model['map2'][(child.tiling, axis)] * cost
        self.add_edge(child_id, self.cur_node_id, e_cost)

      if len(copy_nodes) > 0: self.add_edge(self.cur_node_id, copy_nodes[0], 0)
      copy_nodes.append(self.cur_node_id)
      self.cur_node_id += 1

    e_cost = reduce(operator.mul, expr.shape, 1)
    inter_node_id = copy_nodes[0]
    if len(expr.shape) > 1 and expr.shape[1] > 1:
      child_ids = []
      for tiling_type in range(self.num_node_per_group):
        self.nodes[self.cur_node_id] = self.node_type([expr], tiling_type, [], [])
        self.add_edge(inter_node_id, self.cur_node_id, (tiling_type / 3 + 1) * e_cost)
        child_ids.append(self.cur_node_id)
        self.cur_node_id += 1
      self.add_split_nodes(child_ids)
      return child_ids
    else:
      self.nodes[self.cur_node_id] = self.node_type([expr], 0, [], [])
      self.add_edge(inter_node_id, self.cur_node_id, e_cost)
      self.cur_node_id += 1
      return [self.cur_node_id - 1]

  def visit_OuterProductExpr(self, expr):
    child_id_groups = []
    for array in expr.arrays:
      child_id_groups.append(self.visit_children([array]))

    copy_nodes = []
    for axis, child_ids in zip(expr.axes, child_id_groups):
      self.nodes[self.cur_node_id] = self.node_type([expr], axis, [], [])

      cost = reduce(operator.mul, self.nodes[child_ids[0]].expr[0].shape, 1)
      for child_id in child_ids:
        child = self.nodes[child_id]
        e_cost = 0 if axis is not None and (axis == (child.tiling % 4) or child.tiling == 3) else cost
        self.add_edge(child_id, self.cur_node_id, e_cost)

      if len(copy_nodes) > 0: self.add_edge(self.cur_node_id, copy_nodes[0], cost)
      copy_nodes.append(self.cur_node_id)
      self.cur_node_id += 1

    e_cost = reduce(operator.mul, expr.shape, 1)
    inter_node_id = copy_nodes[0]
    if len(expr.shape) > 1 and expr.shape[1] > 1:
      child_ids = []
      for tiling_type in range(self.num_node_per_group):
        self.nodes[self.cur_node_id] = self.node_type([expr], tiling_type, [], [])
        self.add_edge(inter_node_id, self.cur_node_id, (tiling_type / 3 + 1) * e_cost)
        child_ids.append(self.cur_node_id)
        self.cur_node_id += 1
      self.add_split_nodes(child_ids)
      return child_ids
    else:
      self.nodes[self.cur_node_id] = self.node_type([expr], 0, [], [])
      self.add_edge(inter_node_id, self.cur_node_id, e_cost)
      self.cur_node_id += 1
      return [self.cur_node_id - 1]

  def visit_ShuffleExpr(self, expr):
    for child in expr.fn_kw.itervalues():
      if isinstance(child, (Expr, DistArray)) and hash(child) not in expr.cost_hint:
        cost = reduce(operator.mul, child.shape, 1)
        expr.cost_hint[hash(child)] = {'00': cost, '01': cost, '10': cost, '11': cost}
    if expr.target is not None and hash(expr.target) not in expr.cost_hint:
      cost = reduce(operator.mul, expr.target.shape, 1)
      expr.cost_hint[hash(expr.target)] = {'00': cost, '01': cost, '10': cost, '11': cost}

    child_ids = self.visit_children([expr.array])
    other_child_ids = self.visit_children(expr.fn_kw.itervalues())

    # calc the copy cost
    if len(other_child_ids) == 0:
      expr_node_ids = child_ids
    else:
      if child_ids[0] in self.split_nodes:
        tiling_types = (0, 1)
        self.add_split_nodes(self.cur_node_id, self.cur_node_id + 1)
        expr_node_ids = [self.cur_node_id, self.cur_node_id + 1]
      else:
        tiling_types = (self.nodes[child_ids[0]].tiling,)
        expr_node_ids = [self.cur_node_id]

      for i in xrange(len(tiling_types)):
        tiling_type = tiling_types[i]
        self.nodes[self.cur_node_id] = self.node_type([expr], tiling_type, [], [])
        self.add_edge(child_ids[i], self.cur_node_id, 0)

        for child in expr.fn_kw.itervalues():
          if isinstance(child, (Expr, DistArray)):
            for child_id in self.expr_to_nodes[hash(child)]:
              self.add_edge(child_id, self.cur_node_id, expr.cost_hint[hash(child)]['%d%d' % (self.nodes[child_id].tiling, tiling_type)])
        self.cur_node_id += 1

    # calculate update cost
    if expr.target is not None:
      other_child_ids = expr_node_ids
      child_ids = self.visit_children([expr.target])

      if child_ids[0] in self.split_nodes:
        tiling_types = (0, 1)
        self.add_split_nodes(self.cur_node_id, self.cur_node_id + 1)
        expr_node_ids = [self.cur_node_id, self.cur_node_id + 1]
      else:
        tiling_types = (self.nodes[child_ids[0]].tiling,)
        expr_node_ids = [self.cur_node_id]

      for i in xrange(len(tiling_types)):
        tiling_type = tiling_types[i]
        self.nodes[self.cur_node_id] = self.node_type([expr], tiling_type, [], [])
        self.add_edge(child_ids[i], self.cur_node_id, 0)

        for child_id in other_child_ids:
          self.add_edge(child_id, self.cur_node_id, expr.cost_hint[hash(expr.target)]['%d%d' % (self.nodes[child_id].tiling, tiling_type)])
        self.cur_node_id += 1
      return expr_node_ids

    if expr.shape == expr.array.shape or len(other_child_ids) > 0:
      if len(other_child_ids) == 0:
        for child_id in expr_node_ids: self.nodes[child_id].expr.append(expr)
      return expr_node_ids

    # fix result tiling
    if len(expr.shape) <= 1 or 1 in expr.shape[:2]:
      tiling_type = 0 if len(expr.shape) <= 1 else 1 - expr.shape.index(1)
      self.nodes[self.cur_node_id] = self.node_type([expr], tiling_type, [], [])
      for child_id in expr_node_ids: self.add_edge(child_id, self.cur_node_id, 0)
      self.cur_node_id += 1
      return [self.cur_node_id - 1]
    else:
      child_ids = expr_node_ids
      expr_node_ids = []
      for child_id in child_ids:
        self.nodes[self.cur_node_id] = self.node_type([expr], self.nodes[child_id].tiling, [], [])
        expr_node_ids.append(self.cur_node_id)
        self.add_edge(child_id, self.cur_node_id, 0)
        self.cur_node_id += 1
      if len(expr_node_ids) > 1: self.add_split_nodes(*expr_node_ids)
    return expr_node_ids

  def visit_DotExpr(self, expr):
    child_ids = self.visit_children([expr.matrix_a])
    other_child_ids = self.visit_children([expr.matrix_b])

    # calc copy cost
    if len(other_child_ids) != 0:
      if child_ids[0] in self.split_nodes:
        tiling_types = range(self.num_node_per_group)
        self.add_split_nodes(range(self.cur_node_id, self.cur_node_id + self.num_node_per_group))
        expr_node_ids = range(self.cur_node_id, self.cur_node_id + self.num_node_per_group)
      else:
        tiling_types = (self.nodes[child_ids[0]].tiling,)
        expr_node_ids = [self.cur_node_id]

      for i in xrange(len(tiling_types)):
        tiling_type = tiling_types[i]
        self.nodes[self.cur_node_id] = self.node_type([expr], tiling_type, [], [])
        self.add_edge(child_ids[i], self.cur_node_id, 0)

        for child_id in other_child_ids:
          child = self.nodes[child_id]
          e_cost = 0 if child.tiling in (0, 3) and tiling_type in (1, 2) else reduce(operator.mul, child.expr[0].shape, 1)
          self.add_edge(child_id, self.cur_node_id, e_cost)
        self.cur_node_id += 1

      child_ids = expr_node_ids

    # calc update cost
    if len(expr.shape) == 1 or expr.shape[1] == 1:
      tiling_types = (0,)
      expr_node_ids = [self.cur_node_id]
    else:
      tiling_types = range(self.num_node_per_group)
      self.add_split_nodes(range(self.cur_node_id, self.cur_node_id + self.num_node_per_group))
      expr_node_ids = range(self.cur_node_id, self.cur_node_id + self.num_node_per_group)

    cost = reduce(operator.mul, expr.shape, 1)
    for tiling_type in tiling_types:
      self.nodes[self.cur_node_id] = self.node_type([expr], tiling_type, [], [])
      for child_id in child_ids:
        e_cost = 0 if self.nodes[child_id].tiling in (0, 3) and tiling_type == 0 else cost
        self.add_edge(child_id, self.cur_node_id, e_cost)
      self.cur_node_id += 1
    return expr_node_ids

  def visit_WriteArrayExpr(self, expr):
    child_ids = self.visit_children([expr.array])
    if isinstance(expr.data, (Expr, DistArray)):
      data_child_ids = self.visit_children([expr.data])
      if child_ids[0] in self.split_nodes:
        tiling_types = range(self.num_node_per_group)
        self.add_split_nodes(range(self.cur_node_id, self.cur_node_id + self.num_node_per_group))
        expr_node_ids = range(self.cur_node_id, self.cur_node_id + self.num_node_per_group)
      else:
        tiling_types = (self.nodes[child_ids[0]].tiling,)
        expr_node_ids = [self.cur_node_id]

      for (tiling_type, map_child_id) in zip(tiling_types, child_ids):
        self.nodes[self.cur_node_id] = self.node_type([expr], tiling_type, [], [])
        self.add_edge(map_child_id, self.cur_node_id, 0)

        for child_id in data_child_ids:
          child = self.nodes[child_id]
          e_cost = self.cost_model['map'][(child.tiling, tiling_type)] * reduce(operator.mul, child.expr[0].shape, 1)
          self.add_edge(child_id, self.cur_node_id, e_cost)

        self.cur_node_id += 1
      return expr_node_ids
    else:
      for child_id in child_ids: self.nodes[child_id].expr.append(expr)
      return child_ids

  def visit_aligned_nodes(self, expr, reverse_cost=False):
    array = expr.src if hasattr(expr, 'src') else expr.array
    child_ids = self.visit_children([array])
    if child_ids[0] in self.split_nodes:
      tiling_types = range(self.num_node_per_group)
      self.add_split_nodes(range(self.cur_node_id, self.cur_node_id + self.num_node_per_group))
      expr_node_ids = range(self.cur_node_id, self.cur_node_id + self.num_node_per_group)

      for tiling_type in tiling_types:
        self.nodes[self.cur_node_id] = self.node_type([expr], tiling_type, [], [])
        orig_tiling = -1
        if tiling_type in (2, 3):
          orig_tiling = tiling_type
        elif tiling_type < 2:
          orig_tiling = reverse_cost ^ tiling_type
        else:
          orig_tiling = (reverse_cost ^ (tiling_type % 4)) + 4
        self.add_edge(child_ids[orig_tiling], self.cur_node_id, 0)
        self.cur_node_id += 1
    else:
      child_tiling = self.nodes[child_ids[0]].tiling
      if child_tiling in (2, 3):
        tiling_type = child_tiling
      elif child_tiling < 2:
        tiling_type = reverse_cost ^ child_tiling
      else:
        tiling_type = (reverse_cost ^ (child_tiling % 4)) + 4
      self.nodes[self.cur_node_id] = self.node_type([expr], tiling_type, [], [])
      self.add_edge(child_ids[0], self.cur_node_id, 0)
      expr_node_ids = [self.cur_node_id]
      self.cur_node_id += 1

    return expr_node_ids

  def visit_TransposeExpr(self, expr):
    return self.visit_aligned_nodes(expr, reverse_cost=True)

  def visit_ReshapeExpr(self, expr):
    return self.visit_aligned_nodes(expr, reverse_cost=True)

  def visit_SliceExpr(self, expr):
    return self.visit_aligned_nodes(expr)

  def visit_FilterExpr(self, expr):
    child_ids = self.visit_children([expr.src])
    for child_id in child_ids: self.nodes[child_id].expr.append(expr)
    return child_ids

  def visit_CheckpointExpr(self, expr):
    child_ids = self.visit_children([expr.src])
    for child_id in child_ids: self.nodes[child_id].expr.append(expr)
    return child_ids

  def visit_TileOpExpr(self, expr):
    child_ids = self.visit_children([expr.array])
    for child_id in child_ids: self.nodes[child_id].expr.append(expr)
    return child_ids

  def generate_edges(self, s=0):
    edges = []
    self.nodes[s].parents.sort(key=lambda x: reduce(operator.mul, self.nodes[x].expr[0].shape, 1))
    for parent_id in self.nodes[s].parents:
      edges.append((s, parent_id, self.edges[(s, parent_id)]))
      if parent_id not in self.visited_nodes:
        edges.extend(self.generate_edges(parent_id))
        self.visited_nodes.add(parent_id)
    return edges

  def tile_expr(self, expr, tiling):
    if isinstance(expr, (NdArrayExpr, ReduceExpr, Map2Expr, OuterProductExpr)) and len(expr.shape) > 0:
      expr.tile_hint = list(expr.shape)
      if tiling >= 3:  # duplicate tiling
        print 'dup_tiling', tiling
      elif tiling == 2 and len(expr.shape) > 1:  # block tiling
        expr.tile_hint[0] = int(math.ceil(float(expr.tile_hint[0]) / math.sqrt(FLAGS.num_workers)))
        expr.tile_hint[1] = int(math.ceil(float(expr.tile_hint[1]) / math.sqrt(FLAGS.num_workers)))
        print 'block_tiling', expr.tile_hint, expr.expr_id
      elif len(expr.shape) > tiling:
        expr.tile_hint[tiling] = int(math.ceil(float(expr.tile_hint[tiling]) / FLAGS.num_workers))

  def calc_tiling(self, expr):
    # add T node for graph
    self.nodes[self.cur_node_id] = self.node_type([expr], -1, [], [])
    self.add_edge(self.cur_node_id - 1, self.cur_node_id, 0)
    if self.cur_node_id - 1 in self.split_nodes:
      for i in range(2, self.num_node_per_group + 1):
        self.add_edge(self.cur_node_id - i, self.cur_node_id, 0)
    self.cur_node_id += 1

    # compute best tiling for all exprs
    self.visited_nodes = set()
    edges = self.generate_edges()
    print 'num of groups', len(self.groups)

    nodes = []
    if FLAGS.tiling_alg == 'maxedge':
      nodes = tiling.maxedge_tiling(self.cur_node_id - 1, edges, self.groups)
      print 'maxedge', nodes
    elif FLAGS.tiling_alg == 'mincost':
      nodes = tiling.mincost_tiling(self.cur_node_id - 1, edges, self.groups)
      print 'mincost', nodes
    elif FLAGS.tiling_alg == 'best':
      nodes = tiling.best_tiling(self.cur_node_id - 1, edges, self.groups)
      print 'best', nodes
    elif FLAGS.tiling_alg == 'worse':
      nodes = tiling.worse_tiling(self.cur_node_id - 1, edges, self.groups)
      print 'worse', nodes

    # give expr the best tiling hint
    for node_id in nodes:
      node = self.nodes[node_id]
      if node.tiling < 0 or node.tiling is None: continue
      for cur_expr in node.expr:
        self.tiled_exprlist[hash(cur_expr)] = node.tiling
        self.tile_expr(cur_expr, node.tiling)

    self.inited = False
    return expr

  def tile_cached_expr(self, expr):
    if not isinstance(expr, Expr) or isinstance(expr, (Val, AsArray)): return

    self.tile_expr(expr, self.tiled_exprlist[hash(expr)])

    if hasattr(expr, 'array'): self.tile_cached_expr(expr.array)
    if hasattr(expr, 'src'): self.tile_cached_expr(expr.src)
    if hasattr(expr, 'target'): self.tile_cached_expr(expr.target)
    if isinstance(expr, DotExpr):
      self.tile_cached_expr(expr.matrix_a)
      self.tile_cached_expr(expr.matrix_b)

    if hasattr(expr, 'children'):
      for child in expr.children.vals:
        self.tile_cached_expr(child)
    if hasattr(expr, 'fn_kw') and expr.fn_kw is not None:
      for child in expr.fn_kw.itervalues():
        self.tile_cached_expr(child)
    if hasattr(expr, 'op') and 'fn_kw' in expr.op.kw:
      for child in expr.op.kw['fn_kw'].itervalues():
        self.tile_cached_expr(child)

  def visit_default(self, expr):
    if not self.inited: self.init(expr)
    if hash(expr) in self.tiled_exprlist:
      self.tile_cached_expr(expr)
      tiling = self.tiled_exprlist[hash(expr)]
      self.nodes[self.cur_node_id] = self.node_type([expr], tiling, [], [])
      expr_node_ids = [self.cur_node_id]
      self.add_edge(0, self.cur_node_id, 0)
      self.cur_node_id += 1

    elif hash(expr) in self.expr_to_nodes:
      # cached expr
      expr_node_ids = self.expr_to_nodes[hash(expr)]
      for node_id in expr_node_ids:
        node = self.nodes[node_id]
        node.expr.append(expr)

    elif isinstance(expr, DistArray) or (isinstance(expr, (Val, AsArray)) and isinstance(expr.val, DistArray)):
      # already partitioned array
      array = expr if isinstance(expr, DistArray) else expr.val

      if isinstance(array, LocalWrapper):
        expr_node_ids = []
      else:
        tile_shape = array.tile_shape()
        tiling = 2
        for i in range(len(tile_shape)):
          if tile_shape[i] == array.shape[i]:
            tiling = 1 - i
            break

        self.nodes[self.cur_node_id] = self.node_type([expr], tiling, [], [])
        expr_node_ids = [self.cur_node_id]
        self.add_edge(0, self.cur_node_id, 0)
        self.cur_node_id += 1

    elif isinstance(expr, CollectionExpr):
      # DictExpr, ListExpr, TupleExpr
      children = expr.itervalues() if expr.typename() == 'DictExpr' else expr.vals
      child_ids = self.visit_children(children)
      self.nodes[self.cur_node_id] = self.node_type([expr], -1, [], [])
      expr_node_ids = [self.cur_node_id]
      for child_id in child_ids:
        self.add_edge(child_id, self.cur_node_id, 0)
      self.cur_node_id += 1

    elif hasattr(self, 'visit_%s' % expr.typename()):
      expr_node_ids = getattr(self, 'visit_%s' % expr.typename())(expr)

    else:
      util.log_debug("Skip expr:%s", expr.typename())
      expr_node_ids = []

    # add T node for graph and compute the min cost flow
    if id(expr) == self.init_expr: return self.calc_tiling(expr)

    self.expr_to_nodes[hash(expr)] = expr_node_ids
    return expr_node_ids


def apply_pass(klass, dag):
  if not getattr(FLAGS, 'opt_' + klass.name):
    util.log_debug('Pass %s disabled', klass.name)
    return dag

  util.log_debug('Starting pass %s', klass.name)
  p = klass()
  result = p.visit(dag)
  util.log_debug('Finished pass %s', klass.name)
  return result


passes = []


def optimize(dag):
  if not FLAGS.optimization:
    util.log_debug('Optimizations disabled')
    return dag

  util.log_debug('Optimization: applying %d passes', len(passes))
  for p in passes:
    dag = apply_pass(p, dag)

  return dag


def add_optimization(klass, default):
  passes.append(klass)

  flagname = 'opt_' + klass.name
  #setattr(Flags, flagname, add_bool_flag(flagname, default=default))
  FLAGS.add(BoolFlag(flagname, default=default, help='Enable %s optimization' % klass.__name__))

  #util.log_info('Passes: %s', passes)

add_optimization(CollapsedCachedExpressions, True)
add_optimization(AutomaticTiling, True)
add_optimization(RotateSlice, False)
add_optimization(MapMapFusion, True)
if parakeet is not None:
  add_optimization(ParakeetGeneration, False)
add_optimization(ReduceMapFusion, True)

FLAGS.add(BoolFlag('optimization', default=True))
