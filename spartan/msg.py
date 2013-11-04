#!/usr/bin/env python

"""Message definitions for RPC services."""

from spartan import core
import cPickle
from cloud.serialization import cloudpickle
import numpy as np


class Reduction(object):
  pass

def write(obj, f):
  if isinstance(obj, np.ndarray):
    f.write('N')
    cPickle.dump(obj.dtype, f, protocol= -1)
    cPickle.dump(obj.shape, f, protocol= -1)
    f.write(obj)
  elif isinstance(obj, Message):
    f.write('M')
    f.write_str(obj.__class__.__name__)
    write(obj.__dict__, f)
  elif isinstance(obj, tuple):
    f.write('T')
    f.write_int(len(obj))
    for elem in obj:
      write(elem, f)
  elif isinstance(obj, list):
    f.write('L')
    f.write_int(len(obj))
    for elem in obj:
      write(elem, f)
  elif isinstance(obj, dict):
    f.write('D')
    f.write_int(len(obj))
    for k, v in obj.iteritems():
      write(k, f)
      write(v, f)
  else:
    f.write('P')
    try:
      # print 'Using cpickle for ', obj
      v = cPickle.dumps(obj, -1)
      f.write(v)
    except cPickle.PickleError:
#      print 'Using cloudpickle for ', obj
      cloudpickle.dump(obj, f, protocol= -1)

def read(f):
  datatype = f.read(1)
  if datatype == 'N':
    dtype = cPickle.load(f)
    shape = cPickle.load(f)
    array = N.ndarray(shape, dtype=dtype)
    f.readinto(array)
    return array
  elif datatype == 'M':
    klass = f.read_str()
    klass = eval(klass)
    args = read(f)
    return klass(**args)
  elif datatype == 'T':
    sz = f.read_int()
    return tuple([read(f) for i in range(sz)])
  elif datatype == 'L':
    sz = f.read_int()
    return [read(f) for i in range(sz)]
  elif datatype == 'D':
    sz = f.read_int()
    lst = []
    for i in range(sz):
      k = read(f)
      v = read(f)
      lst.append((k, v))
    return dict(lst)
  elif datatype == 'P':
    res = cPickle.load(f)
    return res

  raise KeyError, 'Unknown datatype: "%s"' % datatype


class Message(object):
  def encode(self, f=None):
    if f is not None:
      return write(self, f)
    f = core.Writer()
    write(self, f)
    return f.getvalue()

  @staticmethod
  def decode(reader):
    return read(reader)

  def copy(self):
    return self.__class__(**self.__dict__)

  def __repr__(self):
    out = '%s { ' % self.__class__.__name__
    for k, v in self.__dict__.iteritems():
      out += '  %s : %s,\n' % (k, v)
    return out + '\n}'

  def __cmp__(self, other):
    return cmp(self.__dict__, other.__dict__)


class Register(Message):
  worker_host = None
  worker_port = None

class Initialize(Message):
  workers = None
  

class DelTable(Message):
  def __init__(self, name):
    self.name = name


class TableDescriptor(Message):
  def __init__(self, name, num_shards, workers=None):
    self.name = name
    self.num_shards = num_shards
    self.workers = workers


class GetRequest(Message):
  def __init__(self, table_id, shard_id, keys=None, whole_shard=None):
    self.table_id = table_id
    self.shard_id = shard_id
    self.keys = keys
    self.whole_shard = whole_shard


class Update(Message):
  def __init__(self, key, op, value):
    self.key = key
    self.op = op
    self.value = value


class Entry(Message):
  def __init__(self, key, value):
    self.key = key
    self.value = value


class GetResponse(Message):
  def __init__(self, entries=None, shard_data=None):
    if entries is None:
      entries = []
    self.entries = entries
    self.shard_data = shard_data


class UpdateRequest(Message):
  def __init__(self, table_id, shard_id, updates=None, shard_data=None):
    self.table_id = table_id
    self.shard_id = shard_id
    if updates is None: updates = []
    self.updates = updates
    self.shard_data = shard_data


class Empty(Message):
  pass

EMPTY = Empty()

class IteratorRequest(Message):
  def __init__(self, table_id, shard_id, iter_id=None):
    self.table_id = table_id
    self.shard_id = shard_id
    self.iter_id = iter_id


class IteratorResponse(Message):
  def __init__(self, iter_id, entries=None):
    self.iter_id = iter_id
    self.entries = entries


class KernelRequest(Message):
  def __init__(self, slices=None, kernel=None):
    if slices is None: slices = []
    self.slices = slices
    self.kernel = kernel


class KernelResponse(Message):
  def __init__(self, results=None):
    if results is None:
      results = []
    self.results = results



class Exception(Message):
  def __init__(self, py_exc):
    self.py_exc = py_exc

