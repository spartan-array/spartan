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

    #util.log_info('New blob ctx.  Worker=%s', self.worker_id)
    
  def is_master(self):
    return self.worker_id == MASTER_ID

  def _send(self, id, method, req, wait=True):
    if self.active == False:
      util.log_debug('Ctx disabled.')
      return None

    #util.log_info('%s %s', id, method)
    worker_id = self._lookup(id)

    # short-circuit for requests to the local worker
    if worker_id == self.worker_id:
      pending_req = rpc.Future(None, -1)
      getattr(self.local_worker, method)(req, pending_req)
    else:
      pending_req = getattr(self.workers[worker_id], method)(req)

    if wait:
      return pending_req.wait()

    return pending_req

  def _send_to_worker(self, worker_id, method, req, wait=True):
    if self.active == False:
      util.log_debug('Ctx disabled.')
      return None

    # short-circuit for requests to the local worker
    if worker_id == self.worker_id:
      pending_req = rpc.Future(None, -1)
      getattr(self.local_worker, method)(req, pending_req)
    else:
      pending_req = getattr(self.workers[worker_id], method)(req)

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

  def destroy_all(self, blob_ids):
    if self.worker_id != MASTER_ID: return
    #util.log_info('Destroy: %s', blob_ids)
    req = core.DestroyReq(ids=blob_ids)

    # fire and forget...?
    self._send_all('destroy', req, wait=False)

  def destroy(self, blob_id):
    return self.destroy_all([blob_id])

  def get(self, blob_id, subslice, callback=None, wait=True):
    Assert.isinstance(blob_id, core.BlobId)
    req = core.GetReq(id=blob_id, subslice=subslice)
#     if self._lookup(blob_id) == self.worker_id:
#       util.log_warn('fake get!')
#     else:
#       util.log_warn('real get!')
    if callback is None:
      if wait:
        return self._send(blob_id, 'get', req).data
      else:
        return self._send(blob_id, 'get', req, wait=False)
    else:
      future = self._send(blob_id, 'get', req, wait=False)
      return future.on_finished(callback)


#   def update(self, blob_id, data, reducer, wait=True):
#     req = core.UpdateReq(id=blob_id, data=data, reducer=reducer)
#     return self._send(blob_id, 'update', req, wait=wait)

  def update(self, blob_id, region, data, reducer, wait=True):

    req = core.UpdateReq(id=blob_id, region=region, data=data, reducer=reducer)
    
#     import scipy.sparse
#     if self._lookup(blob_id) == self.worker_id:
#       util.log_warn('fake update!')
#     elif scipy.sparse.issparse(data) and data.getnnz() == 0:
#       util.log_warn('none update to remote!')
#     else:
#       util.log_warn('update to remote!')
      
    return self._send(blob_id, 'update', req, wait=wait)
  
  def create_local(self):
    assert not self.is_master()
    return core.BlobId(worker=self.worker_id, id=ID_COUNTER.next())

  def create(self, data, hint=None):
    assert self.worker_id >= 0, self.worker_id

    # workers create blobs locally; master dispatches to a 
    # worker in round-robin order.
    if self.is_master():
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

  def partial_map(self, targets, blob_ids, mapper_fn, reduce_fn, kw):
    if self.active == False:
      util.log_debug('Ctx disabled.')
      return None

    req = core.KernelReq(blobs=blob_ids,
                         mapper_fn=mapper_fn,
                         reduce_fn=reduce_fn,
                         kw=kw)

    #util.log_info('%s', req)

    futures = rpc.forall(targets, 'run_kernel', req).wait()
    result = {}
    for f in futures:
      for blob_id, v in f.iteritems():
        result[blob_id] = v
    return result

  def map(self, blob_ids, mapper_fn, reduce_fn, kw):
    return self.partial_map(self.workers.itervalues(),
                            blob_ids,
                            mapper_fn,
                            reduce_fn,
                            kw)

  def tile_op(self, blob_id, fn):
    req = core.TileOpReq(blob_id=blob_id,
                         fn=fn)
    return self._send(blob_id, 'tile_op', req)
  
_ctx = threading.local()


def get():
  return _ctx.val


def set(ctx):
  _ctx.val = ctx

