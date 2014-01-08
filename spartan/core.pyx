#!/usr/bin/env python


import cPickle

import numpy as np

from spartan import util, cloudpickle
from spartan.util import Assert
from spartan.node import Node

from struct import pack, unpack

class Blob(object):
  '''Protocol required for ``Blob`` objects.'''
  def update(self, new_val, reducer):
    pass

  def get(self, subslice):
    pass

cdef class BlobId(object):
  cdef public int worker, id

  def __init__(self, worker, id):
    self.worker = worker
    self.id = id

  def __reduce__(self):
    return (BlobId, (self.worker, self.id))

  def __hash__(BlobId self):
    return self.worker ^ self.id

  def __richcmp__(BlobId self, BlobId other, int op):
    if op == 2:
      return self.worker == other.worker and self.id == other.id
    else:
      raise Exception, 'WTF'

  def __repr__(BlobId self):
    return 'B(%d.%d)' % (self.worker, self.id)



cdef class Message(object):
  def __reduce__(Message self):
    return (self.__class__, tuple(), self.__dict__)

class RegisterReq(Message):
  __metaclass__ = Node
  _members = ['host', 'port']


class RegisterResp(Message):
  __metaclass__ = Node
  pass

class Initialize(Message):
  __metaclass__ = Node
  _members = ['id', 'peers']


class NewBlob(Message):
  __metaclass__ = Node
  _members = ['id', 'data']


class GetReq(Message):
  __metaclass__ = Node
  _members = ['id', 'subslice']


class GetResp(Message):
  __metaclass__ = Node
  _members = ['id', 'data']


class DestroyReq(Message):
  __metaclass__ = Node
  _members = ['ids' ]

class UpdateReq(Message):
  __metaclass__ = Node
  _members = ['id', 'data', 'reducer']


class KernelReq(Message):
  __metaclass__ = Node
  _members = ['blobs', 'mapper_fn', 'reduce_fn', 'kw']


class KernelResp(Message):
  __metaclass__ = Node
  _members = ['result']


class CreateReq(Message):
  __metaclass__ = Node
  _members = ['blob_id', 'data']


class CreateResp(Message):
  __metaclass__ = Node
  _members = ['blob_id']

