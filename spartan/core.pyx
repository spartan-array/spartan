#!/usr/bin/env python


import cPickle

import numpy as np

from spartan import util, cloudpickle
from spartan.util import Assert
from spartan.node import Node, node_type

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

@node_type
class RegisterReq(Message):
  _members = ['host', 'port']


@node_type
class RegisterResp(Message):
  pass

@node_type
class Initialize(Message):
  _members = ['id', 'peers']


@node_type
class NewBlob(Message):
  _members = ['id', 'data']


@node_type
class GetReq(Message):
  _members = ['id', 'subslice']


@node_type
class GetResp(Message):
  _members = ['id', 'data']


@node_type
class DestroyReq(Message):
  _members = ['ids' ]

@node_type
class UpdateReq(Message):
  _members = ['id', 'region', 'data', 'reducer']
  
@node_type
class KernelReq(Message):
  _members = ['blobs', 'mapper_fn', 'reduce_fn', 'kw']


@node_type
class KernelResp(Message):
  _members = ['result']


@node_type
class CreateReq(Message):
  _members = ['blob_id', 'data']


@node_type
class CreateResp(Message):
  _members = ['blob_id']

