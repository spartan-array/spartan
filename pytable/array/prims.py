'''Primitives that backends must support.'''

from .. import util
import pprint

class Primitive(object):
  def __init__(self):
    pass
  
  def to_str(self, indent):
    return self.__class__.__name__ + ' : ' + pprint.pformat(self.__dict__)
  
  def __repr__(self):
    return self.to_str(indent=0)
  
  def __str__(self):
    return self.to_str(indent=0)


class Value(Primitive):
  def __init__(self, value):
    self.value = value
    

class MapTiles(Primitive):
  def __init__(self, array, fn, args):
    self.array = array
    self.fn = fn
    self.args = args


class MapValues(Primitive):
  def __init__(self, array, fn, args):
    assert isinstance(array, Primitive)
    self.array = array
    self.fn = fn
    self.args = args


class NewArray(Primitive):
  def __init__(self, basis, shape, dtype):
    self.basis = basis
    self.shape = shape
    self.dtype = dtype


class ReduceInto(Primitive):
  def __init__(self, src, dst, reducer, args):
    self.args = args
    self.src = src
    self.dst = dst
    self.reducer = reducer


class Join(Primitive):
  def __init__(self, inputs):
    self.inputs = inputs


class Index(Primitive):
  def __init__(self, base, idx):
    self.base = base
    self.idx = idx
