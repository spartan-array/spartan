#!/usr/bin/env python
import threading
import traceback
from spartan import util, rpc
from spartan.util import Assert
from spartan.node import Node

import cloudpickle
import cPickle
import numpy as np


MASTER_ID = 65536


class Message(Node):
  pass


class RegisterReq(Message):
  _members = ['host', 'port']


class RegisterResp(Message):
  pass

class Initialize(Message):
  _members = ['id', 'peers']


class NewBlob(Message):
  _members = ['id', 'data']


class GetReq(Message):
  _members = ['id', 'selector']


class GetResp(Message):
  _members = ['id', 'data']


class UpdateReq(Message):
  _members = ['id', 'data', 'reducer']


class KernelReq(Message):
  _members = ['blobs', 'mapper_fn', 'reduce_fn', 'kw']


class KernelResp(Message):
  _members = ['result']


class CreateReq(Message):
  _members = ['id', 'data']


class CreateResp(Message):
  _members = ['id']


BlobId = int


class Blob(Node):
  _members = ['data', 'id']

  def get(self, selector):
    assert False

  def update(self, value, reducer):
    assert False

id_counter = iter(xrange(10000000))

class BlobCtx(object):
  def __init__(self, worker_id, workers, local_worker=None):
    assert isinstance(workers, dict)
    assert isinstance(worker_id, int)

    self.worker_id = worker_id
    self.workers = workers
    self.id_map = {}
    self.local_worker = local_worker


  def _send(self, id, method, req, wait=True):
    #util.log_info('%s %s', id, method)
    worker_id = self._lookup(id)

    # short-circuit for requests to the local worker
    if worker_id == self.worker_id:
      pending_req = rpc.PendingRequest(None, -1)
      getattr(self.local_worker, method)(req, pending_req)
    else:
      pending_req =  getattr(self.workers[worker_id], method)(req)

    if wait:
      return pending_req.wait()

    return pending_req

  def _lookup(self, blob_id):
    worker_id = blob_id >> 32
    return worker_id
    #if not blob_id in self.id_map:
      #futures = [(id, w.lookup(blob_id)) for id, w in self.workers.iteritems()]
      #for idx, f in enumerate(futures):
      #  if f.wait() is not None:
      #    self.id_map[blob_id] = idx
    #return self.id_map[blob_id]

  def get(self, blob_id, selector):
    Assert.isinstance(blob_id, BlobId)
    req = GetReq(id=blob_id, selector=selector)
    return self._send(blob_id, 'get', req).data

  def update(self, blob_id, data, reducer, wait=True):
    req = UpdateReq(id=blob_id, data=data, reducer=reducer)
    return self._send(blob_id, 'update', req, wait=wait)

  def create(self, data):
    assert self.worker_id >= 0, self.worker_id
    new_id = id_counter.next()

    if self.worker_id >= MASTER_ID:
      worker_id = new_id % len(self.workers)
    else:
      worker_id = self.worker_id

    worker = self.workers[worker_id]
    blob_id = worker_id << 32 | new_id
    #util.log_info('%s %s %s %s', new_id, worker_id, data.shape, ''.join(traceback.format_stack()))

    req = CreateReq(id=blob_id, data=data)
    return self._send(blob_id, 'create', req, wait=False)

  def map(self, blob_ids, mapper_fn, reduce_fn, kw):
    req = KernelReq(blobs=blob_ids,
                    mapper_fn=mapper_fn,
                    reduce_fn=reduce_fn,
                    kw=kw)

    #util.log_info('%s', req)

    futures = [w.run_kernel(req) for w in self.workers.itervalues()]
    result = {}
    for f in futures:
      for blob_id, v in f.wait().iteritems():
        result[blob_id] = v
    return result


_ctx = threading.local()


def get_ctx():
  return _ctx.val


def set_ctx(ctx):
  _ctx.val = ctx

