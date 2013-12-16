#!/usr/bin/env python

'''Local expressions.

Briefly: global expressions are over arrays, and local expressions are over tiles.

`LocalExpr`s have dependencies and can be
chained together; this allows us to construct local DAG's when optimizing,
which can then be executed or converted to parakeet code.
'''
import tempfile
import imp

from spartan import util
from spartan.util import Assert
from spartan.node import Node

var_id = iter(xrange(1000000))


def make_var():
  '''Return a new unique key for use as a variable name'''
  return 'k%d' % var_id.next()


class LocalCtx(object):
  __metaclass__ = Node
  _members = ['inputs', 'axis', 'extent']


class LocalExpr(object):
  '''Represents an internal operation to be performed in the context of a tile.'''

  def add_dep(self, v):
    if self.deps is None: self.deps = []
    self.deps.append(v)

  def codegen(self):
    raise NotImplementedError, self.__class__

  def input_names(self):
    return util.flatten([v.input_names() for v in self.deps], unique=True)


class LocalInput(LocalExpr):
  '''An externally supplied input.'''
  __metaclass__ = Node
  _members = ['idx']

  def __str__(self):
    return 'V(%s)' % self.idx

  def codegen(self):
    return self.idx

  def node_init(self):
    Assert.isinstance(self.idx, str)

  def evaluate(self, ctx):
    return ctx.inputs[self.idx]

  def input_names(self):
    return [self.idx]


class FnCallExpr(LocalExpr):
  _members = ['deps', 'kw', 'fn', 'pretty_fn']

  def node_init(self):
    if self.kw is None: self.kw = {}

  def fn_name(self):
    if self.pretty_fn:
      return self.pretty_fn

    if hasattr(self.fn, '__module__'):
      return '%s.%s' % (self.fn.__module__, self.fn.__name__)
    elif hasattr(self.fn, '__class__'):
      return '%s.%s' % (self.fn.__class__.__module__,
                        self.fn.__name__)
    else:
      return self.fn.__name__


class LocalMapExpr(FnCallExpr):
  __metaclass__ = Node

  def codegen(self):
    name = self.fn_name()

    arg_str = ','.join([v.codegen() for v in self.deps])
    kw_str = ','.join(['%s=%s' % (k, v) for k, v in self.kw.iteritems()])
    if arg_str:
      kw = ',' + kw_str

    return '%s(%s %s)' % (name, arg_str, kw_str)

  def evaluate(self, ctx):
    deps = [d.evaluate(ctx) for d in self.deps]
    return self.fn(*deps, **self.kw)


class LocalReduceExpr(FnCallExpr):
  __metaclass__ = Node

  def node_init(self):
    if self.kw is None: self.kw = {}

  def evaluate(self, ctx):
    deps = [d.evaluate(ctx) for d in self.deps]
    assert len(deps) == 1
    return self.fn(ctx.extent, deps[0], axis=ctx.axis, **self.kw)

  def codegen(self):
    name = self.fn_name()
    arg_str = ','.join([v.codegen() for v in self.deps])
    kw = self.kw.items() + [('axis', 'axis')]
    kw_str = ','.join(['%s=%s' % (k, v) for k, v in kw])
    if arg_str:
      kw = ',' + kw_str

    return '%s(%s %s)' % (name, arg_str, kw_str)


@util.memoize
def _compile_parakeet_source(src):
  '''Compile source code defining a parakeet function.'''
  namespace = {}
  util.log_info('Eval::\n\n%s\n\n', src)
  tf = tempfile.NamedTemporaryFile(suffix='.py', delete=False)
  tf.write(src)
  tf.close()

  try:
    module = imp.load_source('parakeet_temp', tf.name)
  except:
    util.log_info('Failed to build parakeet wrapper')
    raise

  return module._jit_fn


class ParakeetExpr(LocalExpr):
  __metaclass__ = Node
  _members = ['deps', 'source']

  def evaluate(self, ctx):
    names = self.input_names()
    fn = _compile_parakeet_source(self.source)
    kw_args = {}
    for var in names:
      value = ctx.inputs[var]
      kw_args[var] = value

    util.log_info('%s', kw_args)
    return fn(**kw_args)

def codegen(op):
  if isinstance(op, ParakeetExpr):
    return op.source

  fn = 'import parakeet\n'
  fn = fn + 'from spartan import mathlib\n'
  fn = fn + 'import numpy\n'
  fn = fn + '@parakeet.jit\n'
  fn = fn + 'def _jit_fn'
  fn = fn + '(%s):\n  ' % ','.join(op.input_names())
  fn = fn + 'return ' + op.codegen()

  # verify we can compile before proceeding
  _compile_parakeet_source(fn)

  return fn
