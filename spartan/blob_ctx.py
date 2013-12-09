from . import util, rpc, core
import threading
from .util import Assert

MASTER_ID = 65536
ID_COUNTER = iter(xrange(10000000))

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
    self._deferred = []

    #util.log_info('New blob ctx.  Worker=%s', self.worker_id)

  def defer(self, closure):
    self._deferred.append(closure)

  def _send(self, id, method, req, wait=True):
    if self.active == False:
      util.log_debug('Ctx disabled.')
      return None

    # process any deferred operations now:
    if self._deferred:
      fns = self._deferred
      self._deferred = []
      for fn in fns:
        #util.log_info('Running deferred operation %s', fn)
        fn()

    #util.log_info('%s %s', id, method)
    worker_id = self._lookup(id)

    # short-circuit for requests to the local worker
    if worker_id == self.worker_id:
      pending_req = rpc.Future(None, -1)
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
    req = core.DestroyReq(ids=blob_ids)

    # fire and forget...?
    self._send_all('destroy', req, wait=False)

  def destroy(self, blob_id):
    return self.destroy_all([blob_id])

  def get(self, blob_id, selector, callback=None):
    Assert.isinstance(blob_id, core.BlobId)
    req = core.GetReq(id=blob_id, selector=selector)
    if callback is None:
      return self._send(blob_id, 'get', req).data
    else:
      future = self._send(blob_id, 'get', req, wait=False)
      return future.on_finished(callback)


  def update(self, blob_id, data, reducer, wait=True):
    req = core.UpdateReq(id=blob_id, data=data, reducer=reducer)
    return self._send(blob_id, 'update', req, wait=wait)

  def create_local(self):
    assert self.worker_id != MASTER_ID
    return core.BlobId(worker=self.worker_id, id=ID_COUNTER.next())

  def create(self, data, hint=None):
    assert self.worker_id >= 0, self.worker_id

    if self.worker_id >= MASTER_ID:
      if hint is None:
        worker_id = ID_COUNTER.next() % len(self.workers)
      else:
        worker_id = hint % len(self.workers)
      id = -1
    else:
      worker_id = self.worker_id
      id = ID_COUNTER.next()

    blob_id = core.BlobId(worker=worker_id, id=id)

    #util.log_info('%s %s %s %s', new_id, worker_id, data.shape, ''.join(traceback.format_stack()))

    req = core.CreateReq(blob_id=blob_id, data=data)
    return self._send(blob_id, 'create', req, wait=False)

  def map(self, blob_ids, mapper_fn, reduce_fn, kw):
    if self.active == False:
      util.log_debug('Ctx disabled.')
      return None

    req = core.KernelReq(blobs=blob_ids,
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

def get():
  return _ctx.val

def set(ctx):
  _ctx.val = ctx

