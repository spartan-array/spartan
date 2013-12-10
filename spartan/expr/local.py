#!/usr/bin/env python

'''Local expressions.

Briefly: global expressions are over arrays, and local expressions are over tiles.

`LocalExpr`s have dependencies and can be
chained together; this allows us to construct local DAG's when optimizing,
which can then be executed or converted to parakeet code.
'''
from spartan.util import Assert
from spartan.node import Node

var_id = iter(xrange(1000000))


def make_var():
  '''Return a new unique key for use as a variable name'''
  return 'k%d' % var_id.next()


class LocalCtx(object):
  __metaclass__ = Node
  _members = ['inputs', 'axis', 'extent']

  def node_init(self):
    self.code = ''

  def write(self, w):
    self.code = self.code + w

class LocalExpr(object):
  '''Represents an internal operation to be performed in the context of a tile.'''
  def add_dep(self, v):
    if self.deps is None: self.deps = []
    self.deps.append(v)

  def codegen(self, ctx):
    assert False

class LocalInput(LocalExpr):
  '''An externally supplied input.'''
  __metaclass__ = Node
  _members = ['idx']

  def __str__(self):
    return 'V(%s)' % self.idx

  def codegen(self, ctx):
    ctx.write('%s' % self.idx)

  def node_init(self):
    Assert.isinstance(self.idx, str)

  def evaluate(self, ctx):
    return ctx.inputs[self.idx]
    _author__ = 'power'


class LocalMapExpr(LocalExpr):
  _members = ['deps', 'kw', 'fn', 'pretty_fn']
  __metaclass__ = Node

  def node_init(self):
    if self.kw is None: self.kw = {}

  def codegen(self, ctx):
    name = self.pretty_fn if self.pretty_fn else self.fn.__name__
    ctx.write('%s(' % name)
    for v in self.deps:
      v.codegen(ctx)
      ctx.write(',')

    for k, v in self.kw.iteritems():
      ctx.write('%s=%s, ' % (k, v))

    ctx.write(')')

  def evaluate(self, ctx):
    deps = [d.evaluate(ctx) for d in self.deps]
    return self.fn(*deps, **self.kw)


class LocalReduceExpr(LocalExpr):
  __metaclass__ = Node
  _members = ['deps', 'kw', 'fn', 'pretty_fn']

  def node_init(self):
    if self.kw is None: self.kw = {}

  def evaluate(self, ctx):
    deps = [d.evaluate(ctx) for d in self.deps]
    assert len(deps) == 1
    return self.fn(ctx.extent, deps[0], axis=ctx.axis, **self.kw)

  def codegen(self, ctx):
    name = self.pretty_fn if self.pretty_fn else self.fn.__name__
    ctx.write('%s(' % name)
    for v in self.deps:
      v.codegen(ctx)
      ctx.write(',')

    ctx.write('axis=%s' % ctx.axis)
    for k, v in self.kw.iteritems():
      ctx.write('%s=%s, ' % (k, v))
    ctx.write(')')


def codegen(op):
  ctx = LocalCtx()
  op.codegen(ctx)
  return ctx.code

