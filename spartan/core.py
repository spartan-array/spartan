#!/usr/bin/env python
import threading
from spartan import util, rpc
from spartan.util import Assert
from spartan.node import Node

MASTER_ID = 65536


class Message(object):
  pass

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
  _members = ['id', 'selector']


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


class BlobId(Message):
  def __init__(self, worker, id):
    self.worker = worker
    self.id = id

  def __hash__(self): return self.worker ^  self.id
  def __eq__(self, other): return self.worker == other.worker and self.id == other.id
  def __repr__(self): return 'B(%d.%d)' % (self.worker, self.id)

class Blob(object):
  __metaclass__ = Node
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

    self.num_workers = len(workers)
    self.worker_id = worker_id
    self.workers = workers
    self.id_map = {}
    self.local_worker = local_worker
    self.active = True

  def _send(self, id, method, req, wait=True):
    if self.active == False:
      util.log_debug('Ctx disabled.')
      return None

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

  def _send_all(self, method, req, wait=True):
    if self.active == False:
      util.log_debug('Ctx disabled.')
      return None

    futures = rpc.forall(self.workers.itervalues(), method, req)
    if wait:
      return futures.wait()
    return futures

  def _lookup(self, blob_id):
    worker_id = blob_id.worker
    return worker_id
    #if not blob_id in self.id_map:
      #futures = [(id, w.lookup(blob_id)) for id, w in self.workers.iteritems()]
      #for idx, f in enumerate(futures):
      #  if f.wait() is not None:
      #    self.id_map[blob_id] = idx
    #return self.id_map[blob_id]

  def destroy_all(self, blob_ids):
    if self.worker_id != MASTER_ID: return
    #util.log_info('Destroy: %s', blob_ids)
    req = DestroyReq(ids=blob_ids)
    self._send_all('destroy', req)

  def destroy(self, blob_id):
    return self.destroy_all([blob_id])

  def get(self, blob_id, selector):
    Assert.isinstance(blob_id, BlobId)
    req = GetReq(id=blob_id, selector=selector)
    return self._send(blob_id, 'get', req).data

  def update(self, blob_id, data, reducer, wait=True):
    req = UpdateReq(id=blob_id, data=data, reducer=reducer)
    return self._send(blob_id, 'update', req, wait=wait)

  def create_local(self):
    assert self.worker_id != MASTER_ID
    return BlobId(worker=self.worker_id, id=id_counter.next())

  def create(self, data, hint=None):
    assert self.worker_id >= 0, self.worker_id

    if self.worker_id >= MASTER_ID:
      if hint is None:
        worker_id = id_counter.next() % len(self.workers)
      else:
        worker_id = hint % len(self.workers)
      id = -1
    else:
      worker_id = self.worker_id
      id = id_counter.next()

    worker = self.workers[worker_id]
    blob_id = BlobId(worker=worker_id, id=id)
    #util.log_info('%s %s %s %s', new_id, worker_id, data.shape, ''.join(traceback.format_stack()))

    req = CreateReq(blob_id=blob_id, data=data)
    return self._send(blob_id, 'create', req, wait=False)

  def map(self, blob_ids, mapper_fn, reduce_fn, kw):
    req = KernelReq(blobs=blob_ids,
                    mapper_fn=mapper_fn,
                    reduce_fn=reduce_fn,
                    kw=kw)

    #util.log_info('%s', req)

    futures = rpc.forall(self.workers.itervalues(), 'run_kernel', req).wait()
    result = {}
    for f in futures:
      for blob_id, v in f.iteritems():
        result[blob_id] = v
    return result


_ctx = threading.local()


def get_ctx():
  return _ctx.val


def set_ctx(ctx):
  _ctx.val = ctx

