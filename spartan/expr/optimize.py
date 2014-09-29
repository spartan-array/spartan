#!/usr/bin/env python


'''
Optimizations over an expression graph.

Optimization passes take as input an expression graph, and return a
(hopefully) simpler, equivalent graph.  This module defines the
pass infrastructure, the fusion passes and an optimization pass to
lower code to Parakeet.
'''
from collections import namedtuple
import operator, math
from tiling import mincost_tiling
import weakref

from ..config import FLAGS, BoolFlag
from ..array.distarray import DistArray
from . import local
from .filter import FilterExpr
from .slice import SliceExpr
from .local import LocalInput, LocalMapExpr, LocalMapLocationExpr, make_var, ParakeetExpr
from .reduce import ReduceExpr, LocalReduceExpr
from ..util import Assert

from .. import util
from .base import Expr, Val, AsArray, ListExpr, lazify, expr_like, ExprTrace, NotShapeable, CollectionExpr
from .map import MapExpr
from .ndarray import NdArrayExpr
from .shuffle import ShuffleExpr
from .dot import DotExpr
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

    if (not all_maps or isinstance(expr.op, local.ParakeetExpr) or
        id(expr) in _not_idempotent_list):
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
                                op=local.ParakeetExpr(source=source,  deps=expr.op.deps),
                                children=ListExpr(vals = new_children),
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
        if self.visited.get(child_expr, None) != None:
          del self.visited[child_expr]
        child = self.visit(child)

      children.append(child)

    if self.rotated.get(map_expr, None) == True:
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
  inited = False
  
  def init(self, expr):
    self.cur_node_id = 1
    self.edges = {}
    self.nodes = {0: self.node_type([], -1, [], [])}
    self.expr_to_nodes = {}
    self.split_nodes = {}
    self.init_expr = id(expr)
    self.inited = True
      
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
  
  def add_split_nodes(self, node1, node2):
    self.split_nodes[node1] = node2
    self.split_nodes[node2] = node1  
  
  def visit_children(self, children, except_child = None):
    child_ids = []
    for child in children:
      if isinstance(child, (Expr, DistArray)) and id(child) != id(except_child):
        child_ids.extend(self.visit_default(child))          
    return child_ids 
  
  def visit_NdArrayExpr(self, expr):
    # new array need to be partitioned
    if len(expr.shape) > 1 and expr.shape[1] > 1:
      self.nodes[self.cur_node_id] = self.node_type([expr], 0, [], [])
      self.add_edge(0, self.cur_node_id, 0)
      self.cur_node_id += 1    
      
      self.nodes[self.cur_node_id] = self.node_type([expr], 1, [], [])
      self.add_edge(0, self.cur_node_id, 0)
      self.cur_node_id += 1
      
      self.add_split_nodes(self.cur_node_id - 2, self.cur_node_id - 1)
      return [self.cur_node_id - 2, self.cur_node_id - 1]
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
        child = self.nodes[child_id]
        e_cost = reduce(operator.mul, child.expr[0].shape, 1) if child.tiling != tiling_type else 0
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
      e_cost = 0 if expr.axis is None or (1-expr.axis) == child.tiling else cost
      self.add_edge(child_id, self.cur_node_id, e_cost)
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
              self.add_edge(child_id, self.cur_node_id, expr.cost_hint[hash(child)]['%d%d'%(self.nodes[child_id].tiling, tiling_type)])       
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
          self.add_edge(child_id, self.cur_node_id, expr.cost_hint[hash(expr.target)]['%d%d'%(self.nodes[child_id].tiling, tiling_type)])       
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
          child = self.nodes[child_id]
          e_cost = reduce(operator.mul, child.expr[0].shape, 1) if tiling_type == 0 or child.tiling == tiling_type else 0
          self.add_edge(child_id, self.cur_node_id, e_cost)       
        self.cur_node_id += 1
      
      child_ids = expr_node_ids
      
    # calc update cost  
    if len(expr.shape) == 1 or expr.shape[1] == 1:
      tiling_types = (0,)
      expr_node_ids = [self.cur_node_id]
    else:
      tiling_types = (0, 1) 
      self.add_split_nodes(self.cur_node_id, self.cur_node_id + 1)
      expr_node_ids = [self.cur_node_id, self.cur_node_id + 1]
    
    for tiling_type in tiling_types:
      self.nodes[self.cur_node_id] = self.node_type([expr], tiling_type, [], [])
      for child_id in child_ids:
        e_cost = reduce(operator.mul, expr.shape, 1) if self.nodes[child_id].tiling != tiling_type else 0
        self.add_edge(child_id, self.cur_node_id, e_cost)
      self.cur_node_id += 1
    return expr_node_ids
  
  def visit_WriteArrayExpr(self, expr):
    child_ids = self.visit_children([expr.array])
    if isinstance(expr.data, (Expr, DistArray)):
      data_child_ids = self.visit_children([expr.data])
      if child_ids[0] in self.split_nodes:
        tiling_types = (0,1)
        self.add_split_nodes(self.cur_node_id, self.cur_node_id + 1)
        expr_node_ids = [self.cur_node_id, self.cur_node_id + 1]
      else:
        tiling_types = (self.nodes[child_ids[0]].tiling,)
        expr_node_ids = [self.cur_node_id]
        
      for i in xrange(len(tiling_types)):
        self.nodes[self.cur_node_id] = self.node_type([expr], tiling_types[i], [], [])
        self.add_edge(child_ids[i], self.cur_node_id, 0)
        
        for child_id in data_child_ids:
          child = self.nodes[child_id]
          e_cost = reduce(operator.mul, child.expr[0].shape, 1) if child.tiling != tiling_types[i] else 0
          self.add_edge(child_id, self.cur_node_id, e_cost)       
        self.cur_node_id += 1
      return expr_node_ids

    for child_id in child_ids: self.nodes[child_id].expr.append(expr)
    return child_ids 
  
  def visit_aligned_nodes(self, expr, reverse_cost=False):
    array = expr.src if hasattr(expr, 'src') else expr.array
    child_ids = self.visit_children([array])
    if child_ids[0] in self.split_nodes:
      tiling_types = (0, 1)
      self.add_split_nodes(self.cur_node_id, self.cur_node_id + 1)
      expr_node_ids = [self.cur_node_id, self.cur_node_id + 1]
    else:
      tiling_types = (reverse_cost^self.nodes[child_ids[0]].tiling,)
      expr_node_ids = [self.cur_node_id]
    
    for i in xrange(len(tiling_types)):
      self.nodes[self.cur_node_id] = self.node_type([expr], tiling_types[i], [], [])
      self.add_edge(child_ids[-(reverse_cost^i)], self.cur_node_id, 0)       
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
  
  def generate_edges(self, s = 0):
    edges = []
    self.nodes[s].parents.sort(key=lambda x: reduce(operator.mul, self.nodes[x].expr[0].shape, 1))
    for parent_id in self.nodes[s].parents:
      edges.append((s, parent_id, self.edges[(s, parent_id)]))
      if parent_id not in self.visited_nodes:
        edges.extend(self.generate_edges(parent_id))
        self.visited_nodes.add(parent_id)
    return edges
  
  def tile_expr(self, expr, tiling):
    if isinstance(expr, (NdArrayExpr, ReduceExpr, DotExpr)) and len(expr.shape) > 0:
      expr.tile_hint = list(expr.shape)
      expr.tile_hint[tiling] = int(math.ceil(float(expr.tile_hint[tiling]) / FLAGS.num_workers))
 
  def calc_tiling(self, expr):
    # add T node for graph
    self.nodes[self.cur_node_id] = self.node_type([expr], -1, [], [])
    self.add_edge(self.cur_node_id - 1, self.cur_node_id, 0)
    if self.cur_node_id - 1 in self.split_nodes:
      self.add_edge(self.cur_node_id - 2, self.cur_node_id, 0)
    self.cur_node_id += 1
    
    # compute best tiling for all exprs
    self.visited_nodes = set()
    nodes = mincost_tiling(self.cur_node_id - 1, self.generate_edges(), self.split_nodes.items())

    # give expr the best tiling hint
    for node_id in nodes:
      node = self.nodes[node_id]
      for cur_expr in node.expr:
        _tiled_exprlist[hash(cur_expr)] = node.tiling
        self.tile_expr(cur_expr, node.tiling)
      
    self.inited = False
    return expr
  
  def tile_cached_expr(self, expr):
    if not isinstance(expr, Expr) or isinstance(expr, (Val, AsArray)): return
    
    self.tile_expr(expr, _tiled_exprlist[hash(expr)])
    
    if hasattr(expr, 'array'): self.tile_cached_expr(expr.array)
    if hasattr(expr, 'src'): self.tile_cached_expr(expr.src)
    if hasattr(expr, 'target'): self.tile_cached_expr(expr.target)
    if isinstance(expr, DotExpr):
      self.tile_cached_expr(expr.matrix_a)
      self.tile_cached_expr(expr.matrix_b)

    if hasattr(expr, 'children'): 
      for child in expr.children.vals: 
        self.tile_cached_expr(child)
    if hasattr(expr, 'fn_kw'):
      for child in expr.fn_kw.itervalues():
        self.tile_cached_expr(child)
    if hasattr(expr, 'op') and 'fn_kw' in expr.op.kw:
      for child in expr.op.kw['fn_kw'].itervalues():
        self.tile_cached_expr(child)

  def visit_default(self, expr):
    if not self.inited: self.init(expr)
    
    if hash(expr) in _tiled_exprlist:
      self.tile_cached_expr(expr)
      tiling = _tiled_exprlist[hash(expr)]
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

    elif isinstance(expr, DistArray) or isinstance(expr, (Val, AsArray)) and isinstance(expr.val, DistArray):
      # already partitioned array
      array = expr if isinstance(expr, DistArray) else expr.val
      tiling = array.tile_shape()[0] == array.shape[0]
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
  add_optimization(ParakeetGeneration, True)
add_optimization(ReduceMapFusion, True)

FLAGS.add(BoolFlag('optimization', default=True))
