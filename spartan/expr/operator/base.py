'''
Defines the base class of all expressions (`Expr`), as
well as common subclasses for collections.
'''

import collections
import sys
import traceback
import weakref
import numpy as np

from traits.api import Any, Instance, Int, PythonValue

from ... import blob_ctx, util
from ...node import Node, indent
from ...util import Assert, copy_docstring
from ...array import distarray
from ...config import FLAGS, BoolFlag
from ...rpc import TimeoutException

FLAGS.add(BoolFlag('opt_expression_cache', True, 'Enable expression caching.'))


class newaxis(object):
  '''
  The newaxis object indicates that users wish to add a new dimension
  '''


class NotShapeable(Exception):
  '''
  Thrown when the shape for an expression cannot be computed without
  first evaluating the expression.
  '''

unique_id = iter(xrange(10000000))


def _map(*args, **kw):
  '''
  Indirection for handling builtin operators (+,-,/,*).

  (Map is implemented in map.py)
  '''
  fn = kw['fn']

  from .map import map
  return map(args, fn)


def expr_like(expr, **kw):
  '''Construct a new expression like ``expr``.

  The new expression has the same id, but is initialized using ``kw``
  '''
  kw['expr_id'] = expr.expr_id

  trace = kw.pop('trace', None)
  if trace is not None and FLAGS.opt_keep_stack:
    trace.fuse(expr.stack_trace)
    kw['stack_trace'] = trace
  else:
    kw['stack_trace'] = expr.stack_trace

  new_expr = expr.__class__(**kw)

  #util.log_info('Copied %s', new_expr)
  return new_expr


# Pulled out as a class so that we can add documentation.
class EvalCache(object):
  '''
  Expressions can be copied around and changed during optimization
  or due to user actions; we want to ensure that a cache entry can
  be found using any of the equivalent expression nodes.

  To that end, expressions are identfied by an expression id; when
  an expression is copied, the expression ID remains the same.

  The cache tracks results based on expression ID's.  Since this is
  no longer directly linked to an expressions lifetime, we have to
  manually track reference counts here, and clear items from the
  cache when the reference count hits zero.
  '''
  def __init__(self):
    self.refs = collections.defaultdict(int)
    self.cache = {}

  def set(self, exprid, value):
    #assert not exprid in self.cache, 'Trying to replace an existing cache entry!'
    self.cache[exprid] = value

  def get(self, exprid):
    return self.cache.get(exprid, None)

  def register(self, exprid):
    self.refs[exprid] += 1

  def deregister(self, expr_id):
    self.refs[expr_id] -= 1
    if self.refs[expr_id] == 0:
      #util.log_debug('Destroying... %s', expr_id)
      if expr_id in self.cache:
        #import objgraph
        #objgraph.show_backrefs([self.cache[expr_id]], filename='%s-refs.png' % expr_id)
        del self.cache[expr_id]

      del self.refs[expr_id]

  def clear(self):
    self.refs.clear()
    self.cache.clear()


class ExprTrace(object):
  '''
  Captures the stack trace for an expression.

  Lazy evaluation and optimization can result in stack traces that are very far
  from the actual source of an error.  To combat this, expressions track their
  original creation point, which is logged when an error occurs.

  Multiple stack traces can be tracked, as certain optimizations
  will combine multiple expressions together.
  '''
  def __init__(self):
    if FLAGS.capture_expr_stack:
      self.stack_list = [traceback.extract_stack(sys._getframe(3))]
    else:
      self.stack_list = []

  def format_stack(self):
    trace = []
    for i, st in enumerate(self.stack_list):
      trace.append('Stack %d of %d' % (i, len(self.stack_list)))
      trace.append('-' * 80 + '\n')
      for (filename, lineno, fname, txt) in st:
        trace.append('%d >>> %s:%s [%s]: %s\n' % (i, filename, lineno, fname, txt[:60]))
      #trace.extend(st)
    return trace

  def dump(self):
    if not FLAGS.capture_expr_stack:
      print >>sys.stderr, 'Stack tracking for expressions is disabled.  Use --capture_expr_stack=1 to enable.'
      return

    print >>sys.stderr, 'Expr creation stack traceback.'
    if not FLAGS.opt_keep_stack:
      print >>sys.stderr, '    Use --opt_keep_stack=True to see expressions merged during optimization.'

    for s in self.format_stack():
      sys.stderr.write(s)

  def fuse(self, trace):
    self.stack_list.extend(trace.stack_list)


eval_cache = EvalCache()


class Expr(Node):
  '''
  Base class for all expressions.

  `Expr` objects capture user operations.

  An expression can have one or more dependencies, which must
  be evaluated before the expression itself.

  Expressions may be evaluated (using `Expr.evaluate`), the
  result of evaluating an expression is cached until the expression
  itself is reclaimed.
  '''
  expr_id = PythonValue(None, desc="Integer or None")
  stack_trace = Instance(ExprTrace)

  # should evaluation of this object be cached
  needs_cache = True

  optimized_expr = None

  @property
  def ndim(self):
    return len(self.shape)

  def load_data(self, cached_result):
    #util.log_info('expr:%s load_data from not checkpoint node', self.expr_id)
    return None

  def cache(self):
    '''
    Return a cached value for this `Expr`.

    If a cached value is not available, or the cached array is
    invalid (missing tiles), returns None.
    '''
    result = eval_cache.get(self.expr_id)
    if result is not None and len(result.bad_tiles) == 0:
      return result
    return self.load_data(result)

    # get distarray from eval_cache
    # check if still valid
    # if valid, return
    # if not valid: check for disk data
    # if disk data: load bad tiles back
    # else: return None
    #return eval_cache.get(self.expr_id, None)

  def dependencies(self):
    '''
    Returns:
      Dictionary mapping from name to `Expr`.
    '''
    return dict([(k, getattr(self, k)) for k in self.members])

  def compute_shape(self):
    '''
    Compute the shape of this expression.

    If the shape is not available (data dependent), raises `NotShapeable`.

    Returns:
      tuple: Shape of this expression.
    '''
    raise NotShapeable

  def visit(self, visitor):
    '''
    Apply visitor to all children of this node, returning a new `Expr` of the same type.

    :param visitor: `OptimizePass`

    '''
    deps = {}
    for k in self.members:
      deps[k] = visitor.visit(getattr(self, k))

    return expr_like(self, **deps)

  def __repr__(self):
    return self.pretty_str()

  def pretty_str(self):
    '''Return a pretty representation of this node, suitable for showing to users.

    By default, this returns the debug representation.
    '''
    return self.debug_str()
    #raise NotImplementedError, type(self)

  def __del__(self):
    eval_cache.deregister(self.expr_id)

  def __init__(self, *args, **kw):
    super(Expr, self).__init__(*args, **kw)
    #assert self.expr_id is not None
    if self.expr_id is None:
      self.expr_id = unique_id.next()
    else:
      Assert.isinstance(self.expr_id, int)

    if self.stack_trace is None:
      self.stack_trace = ExprTrace()

    eval_cache.register(self.expr_id)
    self.needs_cache = self.needs_cache and FLAGS.opt_expression_cache

  def evaluate(self):
    '''
    Evaluate an `Expr`.

    Dependencies are evaluated prior to evaluating the expression.
    The result of the evaluation is stored in the expression cache,
    future calls to evaluate will return the cached value.

    Returns:
      DistArray:
    '''

    cache = self.cache()
    if cache is not None:
      util.log_debug('Retrieving %d from cache' % self.expr_id)
      return cache

    ctx = blob_ctx.get()
    #util.log_info('Evaluting deps for %s', prim)
    deps = {}
    for k, vs in self.dependencies().iteritems():
      if isinstance(vs, Expr):
        deps[k] = vs.evaluate()
      else:
        #assert not isinstance(vs, (dict, list)), vs
        deps[k] = vs
    try:
      value = self._evaluate(ctx, deps)
      #value = self.optimized()._evaluate(ctx, deps)
    except TimeoutException:
      util.log_info('%s %d need to retry', self.__class__, self.expr_id)
      return self.evaluate()
    except Exception:
      print >>sys.stderr, 'Error executing expression'
      self.stack_trace.dump()
      raise

    if self.needs_cache:
      #util.log_info('Caching %s -> %s', prim.expr_id, value)
      eval_cache.set(self.expr_id, value)

    return value

  def _evaluate(self, ctx, deps):
    '''
    Evaluate this expression.

    Args:
      ctx: `BlobCtx` for interacting with the cluster
      deps (dict): Map from name to `DistArray` or scalar.
    '''
    raise NotImplementedError

  def __hash__(self):
    return self.expr_id

  def typename(self):
    return self.__class__.__name__

  def __add__(self, other):
    return _map(self, other, fn=np.add)

  def __sub__(self, other):
    return _map(self, other, fn=np.subtract)

  def __mul__(self, other):
    '''
    Multiply 2 expressions.

    :param other: `Expr`
    '''
    return _map(self, other, fn=np.multiply)

  def __mod__(self, other):
    return _map(self, other, fn=np.mod)

  def __div__(self, other):
    return _map(self, other, fn=np.divide)

  def __eq__(self, other):
    return _map(self, other, fn=np.equal)

  def __ne__(self, other):
    return _map(self, other, fn=np.not_equal)

  def __lt__(self, other):
    return _map(self, other, fn=np.less)

  def __gt__(self, other):
    return _map(self, other, fn=np.greater)

  def __and__(self, other):
    return _map(self, other, fn=np.logical_and)

  def __or__(self, other):
    return _map(self, other, fn=np.logical_or)

  def __xor(self, other):
    return _map(self, other, fn=np.logical_xor)

  def __pow__(self, other):
    return _map(self, other, fn=np.power)

  def __neg__(self):
    return _map(self, fn=np.negative)

  def __rsub__(self, other):
    return _map(other, self, fn=np.subtract)

  def __radd__(self, other):
    return _map(other, self, fn=np.add)

  def __rmul__(self, other):
    return _map(other, self, fn=np.multiply)

  def __rdiv__(self, other):
    return _map(other, self, fn=np.divide)

  def reshape(self, new_shape):
    '''
    Return a new array with shape``new_shape``, and data from
    this array.

    :param new_shape: `tuple` with same total size as original shape.

    '''
    from . import reshape
    return reshape(self, new_shape)

  def __getitem__(self, idx):
    from .slice import SliceExpr
    from .filter import FilterExpr
    from .reshape import ReshapeExpr

    if isinstance(idx, (int, tuple, slice)):
      is_del_dim = False
      del_dim = list()
      if isinstance(idx, tuple):
        for x in xrange(len(idx)):
          if isinstance(idx[x], int):
            is_del_dim = True
            del_dim.append(x)

      if isinstance(idx, int) or is_del_dim or (isinstance(idx, tuple) and (newaxis in idx)):
        #The shape has to be updated
        if isinstance(idx, tuple):
          new_shape = tuple([slice(x, None, None) if x == -1 else x for x in idx if not x == newaxis])
        else:
          new_shape = idx
        ret = SliceExpr(src=self, idx=new_shape)

        new_shape = []
        if isinstance(idx, tuple):
          shape_ptr = idx_ptr = 0
          while shape_ptr < len(ret.shape) or idx_ptr < len(idx):
            if idx_ptr < len(idx) and idx[idx_ptr] == newaxis:
              new_shape.append(1)
            else:
              new_shape.append(ret.shape[shape_ptr])
              shape_ptr += 1
            idx_ptr += 1
        else:
          new_shape = list(ret.shape)
          del_dim.append(0)

        #Delete dimension if needed
        if is_del_dim:
          for i in del_dim:
            new_shape.pop(i)
        return ReshapeExpr(array=ret, new_shape=new_shape)
      else:
        #This means it's just a simple slice op
        return SliceExpr(src=self, idx=idx)

    else:
      return FilterExpr(src=self, idx=idx)

  def __setitem__(self, k, val):
    raise Exception, 'Expressions are read-only.'

  @property
  def shape(self):
    '''Try to compute the shape of this expression.

    If the value has been computed already this always succeeds.

    :rtype: `tuple`

    '''
    cache = self.cache()
    if cache is not None:
      return cache.shape

    try:
      return self.compute_shape()
    except NotShapeable:
      util.log_debug('Not shapeable: %s', self)
      return evaluate(self).shape

  @property
  def size(self):
    return np.prod(self.shape)

  def optimized(self):
    '''
    Return an optimized version of this expression graph.

    :rtype: `Expr`

    '''
    # If the expr has been optimized, return the cached optimized expr.
    #return optimized_dag(self)

    if self.optimized_expr is None:
      self.optimized_expr = optimized_dag(self)
      self.optimized_expr.optimized_expr = self.optimized_expr
      return self.optimized_expr
    else:
      return self.optimized_expr

  def glom(self):
    '''
    Evaluate this expression and convert the resulting
    distributed array into a Numpy array.

    :rtype: `np.ndarray`

    '''
    return glom(self)

  def __reduce__(self):
    return evaluate(self).__reduce__()


class AsArray(Expr):
  '''Promote a value to be array-like.

  This should be wrapped around most user-inputs that may be
  used in an array context, e.g. (``1 + x => map((as_array(1), as_array(x)), +)``)
  '''
  val = PythonValue

  def visit(self, visitor):
    return self

  def compute_shape(self):
    if hasattr(self.val, 'shape'):
      return self.val.shape

    if np.isscalar(self.val):
      return np.asarray(self.val).shape

    raise NotShapeable

  def _evaluate(self, ctx, deps):
    util.log_debug('%s: Array promotion: value=%s', self.expr_id, deps['val'])
    return distarray.as_array(deps['val'])

  def pretty_str(self):
    return str(self.val)


class Val(Expr):
  '''Convert an existing value to an expression.'''
  val = PythonValue

  needs_cache = False

  @copy_docstring(Expr.visit)
  def visit(self, visitor):
    return self

  def dependencies(self):
    return {}

  @copy_docstring(Expr.visit)
  def compute_shape(self):
    return self.val.shape

  def _evaluate(self, ctx, deps):
    return self.val

  def pretty_str(self):
    return str(self.val)


class CollectionExpr(Expr):
  '''
  `CollectionExpr` subclasses wrap normal tuples, lists and dicts with `Expr` semantics.

  `CollectionExpr.visit` and `CollectionExpr.evaluate` will visit or evaluate
  all of the tuple, list or dictionary elements in this expression.
  '''
  needs_cache = False
  vals = PythonValue

  def _evaluate(self, ctx, deps):
    return deps

  def __getitem__(self, idx):
    return self.vals[idx]

  def __iter__(self):
    return iter(self.vals)


class DictExpr(CollectionExpr):
  def iteritems(self):
    return self.vals.iteritems()

  def keys(self):
    return self.vals.keys()

  def values(self):
    return self.vals.values()

  def itervalues(self):
    return self.vals.itervalues()

  def iterkeys(self):
    return self.vals.iterkeys()

  def pretty_str(self):
    return '{ %s } ' % ',\n'.join(
        ['%s : %s' % (k, repr(v)) for k, v in self.vals.iteritems()])

  def dependencies(self):
    return self.vals

  def visit(self, visitor):
    return DictExpr(vals=dict([(k, visitor.visit(v)) for (k, v) in self.vals.iteritems()]))


class ListExpr(CollectionExpr):
  def dependencies(self):
    return dict(('v%d' % i, self.vals[i]) for i in range(len(self.vals)))

  def pretty_str(self):
    return indent('[\n%s\n]') % ','.join([v.pretty_str() for v in self.vals])

  def __str__(self):
    return repr(self)

  def _evaluate(self, ctx, deps):
    ret = []
    for i in range(len(self.vals)):
      k = 'v%d' % i
      ret.append(deps[k])
    return ret

  def visit(self, visitor):
    return ListExpr(vals=[visitor.visit(v) for v in self.vals])

  def __len__(self):
    return len(self.vals)


class TupleExpr(CollectionExpr):
  def dependencies(self):
    return dict(('v%d' % i, self.vals[i]) for i in range(len(self.vals)))

  def pretty_str(self):
    return '( %s )' % ','.join([v.pretty_str() for v in self.vals])

  def _evaluate(self, ctx, deps):
    ret = []
    for i in range(len(self.vals)):
      k = 'v%d' % i
      ret.append(deps[k])
    return tuple(ret)

  def visit(self, visitor):
    return TupleExpr(vals=tuple([visitor.visit(v) for v in self.vals]))

  def __len__(self):
    return len(self.vals)


def glom(value):
  '''
  Evaluate this expression and return the result as a `numpy.ndarray`.
  '''
  if isinstance(value, Expr):
    value = evaluate(value)

  if isinstance(value, np.ndarray):
    return value

  return value.glom()


def optimized_dag(node):
  '''
  Optimize and return the DAG representing this expression.

  :param node: The node to compute a DAG for.
  '''
  if not isinstance(node, Expr):
    raise TypeError

  from . import optimize
  return optimize.optimize(node)


def evaluate(node):
  '''
  Evaluate ``node``.

  :param node: `Expr`
  '''
  if isinstance(node, Expr):
    return node.evaluate()

  Assert.isinstance(node, (np.ndarray, distarray.DistArray))
  return node


def eager(node):
  '''
  Eagerly evaluate ``node`` and convert the result back into an `Expr`.

  :param node: `Expr` to evaluate.
  '''
  result = evaluate(node)
  return Val(val=result)


def lazify(val):
  '''
  Lift ``val`` into an Expr node.

  If ``val`` is already an expression, it is returned unmodified.

  :param val: anything.
  '''
  #util.log_info('Lazifying... %s', val)
  if isinstance(val, Expr):
    return val

  if isinstance(val, dict):
    return DictExpr(vals=val)

  if isinstance(val, list):
    return ListExpr(vals=val)

  if isinstance(val, tuple):
    return TupleExpr(vals=val)

  return Val(val=val)


def as_array(v):
  '''
  Convert a numpy value or scalar into an `Expr`.

  :param v: `Expr`, numpy value or scalar.
  '''
  if isinstance(v, Expr):
    return v
  else:
    return AsArray(val=v)
