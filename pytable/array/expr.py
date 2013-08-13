'''Lazy arrays.

Expr operations are not performed immediately, but are set aside
and built into a control flow graph, which is then compiled
into a series of primitive array operations.
'''

import numpy as N
import types

class Expr(object):
  def __add__(self, other):
    return Op(op=N.add, children=(self, other))

  def __sub__(self, other):
    return Op(op=N.subtract, children=(self, other))

  def __mul__(self, other):
    return Op(op=N.multiply, children=(self, other))

  def __mod__(self, other):
    return Op(op=N.mod, children=(self, other))

  def __div__(self, other):
    return Op(op=N.divide, children=(self, other))

  def __eq__(self, other):
    return Op(op=N.equal, children=(self, other))

  def __ne__(self, other):
    return Op(op=N.not_equal, children=(self, other))

  def __lt__(self, other):
    return Op(op=N.less, children=(self, other))

  def __gt__(self, other):
    return Op(op=N.greater, children=(self, other))

  def __pow__(self, other):
    return Op(op=N.power, children=(self, other))

  def __getitem__(self, idx):
    return Op('index', children=(self, idx))

  def __setitem__(self, k, val):
    raise Exception, '__setitem__ not supported.'

Expr.__rsub__ = Expr.__sub__
Expr.__radd__ = Expr.__add__
Expr.__rmul__ = Expr.__mul__
Expr.__rdiv__ = Expr.__div__

class LazyVal(Expr):
  def __init__(self, val):
    self._val = val

  def __repr__(self):
    return 'Lazy(%s)' % id(self._val)


def lazify(val):
  if isinstance(val, Expr): return val
  return LazyVal(val)

def val(x):
  return lazify(x)

def pretty(op):
  if isinstance(op, (types.FunctionType,
                     types.BuiltinFunctionType,
                     types.MethodType,
                     types.BuiltinMethodType)):
    return op.__name__
  return repr(op)

class Op(Expr):
  def __init__(self, op, children, kwargs=None):
    """Represents a numpy expression.

    :param op: The operation to perform.
    :param children: Child expressions which will become arguments to ``op``.
    :param args:
    :param kwargs:
    """

    if not isinstance(children, tuple):
      children = (children,)

    self.op = op
    self.children = [lazify(child) for child in children]
    self.kwargs = kwargs

  def __repr__(self):
    return '%s(%s){%s}' % (self.__class__.__name__,
                           pretty(self.op),
                           ','.join([repr(c) for c in self.children]))

  def _dtype(self):
    return self.args[0].dtype

def outer(a, b):
  return Op(N.outer, (a, b))

def dot(a, b):
  return Op(N.dot, (a, b))

def sum(x, axis=None):
  return Op(N.sum, (x,), kwargs={'axis' : axis })

def argmin(x, axis=None):
  return Op(N.argmin, (x,), kwargs={'axis' : axis })

def size(x, axis=None):
  return Op('size', (x,), kwargs={'axis' : axis })

def mean(x, axis=None):
  return sum(x, axis) / size(x, axis)

def astype(x, dtype):
  assert x is not None
  return Op('astype', (x,), kwargs={ 'dtype' : dtype })

def ravel(v):
  return Op(N.ravel, (v,))

def diag(v):
  return Op(N.diag, (v,))

def diagflat(v):
  return diag(ravel(v))

Expr.outer = outer
Expr.dot = dot
Expr.sum = sum
Expr.mean = mean
Expr.astype = astype
Expr.diag = diag
Expr.diagflat = diagflat
Expr.ravel = ravel
Expr.argmin = argmin
