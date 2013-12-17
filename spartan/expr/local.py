#!/usr/bin/env python

'''Local expressions.

Briefly: global expressions are over arrays, and local expressions are over tiles.

`LocalExpr`s have dependencies and can be
chained together; this allows us to construct local DAG's when optimizing,
which can then be executed or converted to parakeet code.
'''
import tempfile
import imp
import types

from spartan import util
from spartan.util import Assert
from spartan.node import Node

var_id = iter(xrange(1000000))

class CodegenException(Exception): pass

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

  def input_names(self):
    return util.flatten([v.input_names() for v in self.deps], unique=True)


class LocalInput(LocalExpr):
  '''An externally supplied input.'''
  __metaclass__ = Node
  _members = ['idx']

  def __str__(self):
    return 'V(%s)' % self.idx

  def node_init(self):
    Assert.isinstance(self.idx, str)

  def evaluate(self, ctx):
    return ctx.inputs[self.idx]

  def input_names(self):
    return [self.idx]


class FnCallExpr(LocalExpr):
  __metaclass__ = Node
  _members = ['deps', 'kw', 'fn', 'pretty_fn']

  def node_init(self):
    if self.kw is None: self.kw = {}
    assert self.fn is not None

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

  def evaluate(self, ctx):
    deps = [d.evaluate(ctx) for d in self.deps]
    #util.log_info('%s %s', deps, self.kw)
    return self.fn(*deps, **self.kw)


# The local operation of map and reduce expressions is practically
# identical.  Reductions take an axis and extent argument in
# addition to the normal function call arguments.
class LocalMapExpr(FnCallExpr):
  _op_type = 'map'

class LocalReduceExpr(FnCallExpr):
  _op_type = 'reduce'


@util.memoize
def compile_parakeet_source(src):
  '''Compile source code defining a parakeet function.'''
  namespace = {}
  util.log_info('Eval::\n\n%s\n\n', src)
  tf = tempfile.NamedTemporaryFile(suffix='.py', delete=False)
  tf.write(src)
  tf.close()

  import atexit
  import os
  atexit.register(lambda: os.remove(tf.name))

  try:
    module = imp.load_source('parakeet_temp', tf.name)
  except Exception, ex:
    util.log_info('Failed to build parakeet wrapper.  Source was: %s', src)
    raise CodegenException(ex.message, ex.args)
  return module._jit_fn


class ParakeetExpr(LocalExpr):
  __metaclass__ = Node
  _members = ['deps', 'source']

  def evaluate(self, ctx):
    names = self.input_names()
    fn = compile_parakeet_source(self.source)
    kw_args = {}
    for var in names:
      value = ctx.inputs[var]
      kw_args[var] = value

    util.log_info('%s', kw_args)
    return fn(**kw_args)

