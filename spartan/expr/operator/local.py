#!/usr/bin/env python

'''Local expressions.

Briefly: global expressions are over arrays, and local expressions are over tiles.

`LocalExpr`s have dependencies and can be
chained together; this allows us to construct local DAG's when optimizing,
which can then be executed or converted to parakeet code.
'''
import imp
import tempfile
import time

import numpy as np
import scipy.sparse as sp

from spartan import util
from spartan.util import Assert
from spartan.node import Node, indent
from traits.api import Str, List, Function, PythonValue, Int

var_id = iter(xrange(1000000))
expr_id = iter(xrange(1000000))


class CodegenException(Exception): pass


def make_var():
  '''Return a new unique key for use as a variable name'''
  return 'key_%d' % var_id.next()


class LocalCtx(Node):
  inputs = PythonValue


class LocalExpr(Node):
  '''Represents an internal operation to be performed in the context of a tile.'''
  deps = List()

  def __init__(self, *args, **kw):
    super(LocalExpr, self).__init__(*args, **kw)

  def __repr__(self):
    # return self.debug_str()
    return self.pretty_str()

  def add_dep(self, v):
    self.deps.append(v)
    #assert len(self.deps) <= 2, v

  def input_names(self):
    return util.flatten([v.input_names() for v in self.deps], unique=True)


class LocalInput(LocalExpr):
  '''An externally supplied input.'''
  idx = Str()

  def __init__(self, *args, **kw):
    LocalExpr.__init__(self, *args, **kw)
    assert self.idx != ''

  def pretty_str(self):
    return '%s' % self.idx

  def evaluate(self, ctx):
    return ctx.inputs[self.idx]

  def input_names(self):
    return [self.idx]


class FnCallExpr(LocalExpr):
  '''Evaluate a function call.

  Dependencies that are variable should be specified via the ``deps`` attribute,
  and will be evaluated and supplied to the function when called.

  Constants (axis of a reduction, datatype, etc), can be supplied via the ``kw``
  argument.
  '''
  kw = PythonValue
  fn = PythonValue
  pretty_fn = PythonValue

  def __init__(self, *args, **kw):
    super(FnCallExpr, self).__init__(*args, **kw)
    if self.kw is None: self.kw = {}
    assert self.fn is not None

  def pretty_str(self):
    # drop modules from the prettified string
    pretty_fn = self.fn_name().split('.')[-1]
    return '%s(%s,kw=%s)' % (
      pretty_fn, indent(','.join([v.pretty_str() for v in self.deps if not isinstance(v, LocalInput)])),
      indent(','.join(['(k=%s v=%s)' % (k, v) for k, v in self.kw.iteritems()]))
    )

  def fn_name(self):
    '''Return a name for this function suitable for calling.'''
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

    #util.log_info('Evaluating %s.%d [%s]', self.fn_name(), self.id, deps)

    # Not all Numpy operations are compatible with mixed sparse and dense arrays.
    # To address this, if only one of the inputs is sparse, we convert it to
    # dense before computing our result.
    if isinstance(self.fn, np.ufunc) and len(deps) == 2 and sp.issparse(deps[0]) ^ sp.issparse(deps[1]):
      for i in range(2):
        if sp.issparse(deps[i]):
          deps[i] = deps[i].todense()
    return self.fn(*deps, **self.kw)


# The local operation of map and reduce expressions is practically
# identical.  Reductions take an axis and extent argument in
# addition to the normal function call arguments.
class LocalMapExpr(FnCallExpr):
  _op_type = 'map'


class LocalMapLocationExpr(LocalMapExpr):
  _op_type = 'map_location'

  def evaluate(self, ctx):
    deps = []
    for d in self.deps:
      if isinstance(d, LocalInput) and d.idx == 'extent':
        deps.append(d.evaluate(ctx).to_tuple())
      else:
        deps.append(d.evaluate(ctx))

    #util.log_info('Evaluating %s.%d [%s]', self.fn_name(), self.id, deps)
    return self.fn(*deps, **self.kw)


class LocalReduceExpr(FnCallExpr):
  _op_type = 'reduce'

# track source that we have already compiled via parakeet.
# parakeet requires the source file remain available in
# order to compile.
source_files = []


# memoize generated modules to avoid recompiling parakeet
# functions for the same source.
@util.memoize
def compile_parakeet_source(src):
  '''Compile source code defining a parakeet function.'''
  util.log_debug('Compiling parakeet source.')
  tmpfile = tempfile.NamedTemporaryFile(delete=True, prefix='spartan-local-', suffix='.py')
  tmpfile.write(src)
  tmpfile.flush()

  #util.log_info('File: %s, Source: \n %s \n', tmpfile.name, src)

  #os.rename(tmpfile.name, srcfile)
  #atexit.register(lambda: os.remove(srcfile))

  try:
    module = imp.load_source('parakeet_temp', tmpfile.name)
  except Exception, ex:
    util.log_info('Failed to build parakeet wrapper')
    util.log_debug('Source was: %s', src)
    raise CodegenException(ex.message, ex.args)

  source_files.append(tmpfile)
  return module._jit_fn


class ParakeetExpr(LocalExpr):
  deps = PythonValue
  source = PythonValue

  def fn_name(self):
    return 'parakeet'

  def pretty_str(self):
    return 'parakeet_op'

  def evaluate(self, ctx):
    names = self.input_names()
    fn = compile_parakeet_source(self.source)

    kw_args = {}
    for var in names:
      value = ctx.inputs[var]
      kw_args[var] = value

    if FLAGS.use_cuda:
      return fn(_backend='cuda', **kw_args)
    else:
      return fn(**kw_args)

from spartan.config import FLAGS, BoolFlag
FLAGS.add(BoolFlag('use_cuda', default=False))
