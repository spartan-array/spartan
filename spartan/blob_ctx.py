'''
The `BlobCtx` manages the state of a Spartan execution: it stores the 
location of tiles and other workers in the system, and contains methods
for fetching and updating array data, creating and removing tiles and
running user-defined *kernel functions* on tile data.
'''


from . import util, rpc, core
import threading
from .util import Assert
import random

MASTER_ID = 65536
ID_COUNTER = iter(xrange(10000000))


class BlobCtx(object):
  def __init__(self, worker_id, workers, local_worker=None):
    '''
    Create a new context.
    
    Args:
      worker_id (int): Identifier for this worker
      workers (list of RPC clients): RPC connections to other workers in the computation.
      local_worker (Worker): A reference to the local worker creating this context.
        This is used to avoid sending RPC messages for operations that can be 
        serviced locally.
    '''
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
    '''
    True if this context is running in the master process.
    '''
    return self.worker_id == MASTER_ID

  def _send(self, id, method, req, wait=True, timeout=None):
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
      pending_req = getattr(self.workers[worker_id], method)(req, timeout)

    if wait:
      return pending_req.wait()

    return pending_req

  def _send_to_worker(self, worker_id, method, req, wait=True, timeout=None):
    if self.active == False:
      util.log_debug('Ctx disabled.')
      return None

    # short-circuit for requests to the local worker
    if worker_id == self.worker_id:
      pending_req = rpc.Future(None, -1)
      getattr(self.local_worker, method)(req, pending_req)
    else:
      pending_req = getattr(self.workers[worker_id], method)(req, timeout)

    if wait:
      return pending_req.wait()

    return pending_req
  
  def _send_all(self,  method, req, targets=None, wait=True, timeout=None):
    if self.active == False:
      util.log_debug('Ctx disabled.')
      return None

    if targets is None:
      available_workers = self.local_worker.get_available_workers()
      targets = [self.workers[worker_id] for worker_id in available_workers]
      
    futures = rpc.forall(targets, method, req, timeout)
    if wait:
      return futures.wait()
    return futures

  def _lookup(self, tile_id):
    worker_id = tile_id.worker
    return worker_id

  def destroy_all(self, tile_ids):
    '''
    Destroy all tiles 
    
    Args:
      tile_ids (list): Tiles to destroy. 
    '''
    Assert.eq(self.worker_id, MASTER_ID)
    
    #util.log_info('Destroy: %s', tile_ids)
    req = core.DestroyReq(ids=tile_ids)
    
    # Don't need to wait for the result.
    self._send_all('destroy', req, wait=False)

  def destroy(self, tile_id):
    '''
    Destroy a tile.
    
    Args:
      tile_id (int): Tile to destroy.
    '''
    return self.destroy_all([tile_id])

  def get(self, tile_id, subslice, wait=True, timeout=None):
    '''
    Fetch a region of a tile.
    
    Args:
      tile_id (int): Tile to fetch from.
      subslice (slice or None): Portion of tile to fetch.
      wait (boolean): Wait for this operation to finish before returning.
      timeout (float):
    '''
    Assert.isinstance(tile_id, core.TileId)
    req = core.GetReq(id=tile_id, subslice=subslice)

    if wait:
      return self._send(tile_id, 'get', req, wait=True, timeout=timeout).data
    else:
      return self._send(tile_id, 'get', req, wait=False)


#   def update(self, tile_id, data, reducer, wait=True):
#     req = core.UpdateReq(id=tile_id, data=data, reducer=reducer)
#     return self._send(tile_id, 'update', req, wait=wait)

  def update(self, tile_id, region, data, reducer, wait=True, timeout=None):
    '''
    Update ``region`` of ``tile_id`` with ``data``.
    
    ``data`` is combined with existing tile data using ``reducer``.
    
    Args:
      tile_id (int):
      region (slice):
      data (Numpy array):
      reducer (function): function from (array, array) -> array
      wait (boolean): If true, wait for completion before returning.
      timeout (float):
    '''

    req = core.UpdateReq(id=tile_id, region=region, data=data, reducer=reducer)
    return self._send(tile_id, 'update', req, wait=wait, timeout=timeout)
  
  def new_tile_id(self):
    '''
    Create a new tile id.  Does not create a new tile, or any data. 
    
    Returns:
      `TileId`: Id of created tile.
    '''
    assert not self.is_master()
    return core.TileId(worker=self.worker_id, id=ID_COUNTER.next())
  
  def heartbeat(self, worker_status, timeout=None):
    '''
    Send a heartbeat request to the master.
    
    :param worker_status:
    :param timeout:
    '''
    req = core.HeartbeatReq(worker_id=self.worker_id, worker_status=worker_status)
    return self._send_to_worker(MASTER_ID, 'heartbeat', req, wait=False, timeout=timeout)
  
  def create(self, data, hint=None, timeout=None):
    '''
    Create a new tile to hold ``data``.
    
    Args:
      data (Numpy array): Data to store in a tile.
      hint (int): Optional.  Worker to store data on.  
        If not specified, workers are chosen in round-robin order.
      timeout (float):
    '''
    assert self.worker_id >= 0, self.worker_id

    # workers create blobs locally; master dispatches to a 
    # worker in round-robin order.
    if self.is_master():
      if hint is None:
        worker_id = ID_COUNTER.next() % len(self.workers)
      else:
        worker_id = hint % len(self.workers)
        
      if worker_id not in self.local_worker._available_workers:
        worker_id = random.choice(self.local_worker._available_workers)
        
      id = -1
    else:
      worker_id = self.worker_id
      id = ID_COUNTER.next()

    tile_id = core.TileId(worker=worker_id, id=id)

    #util.log_info('%s %s %s %s', new_id, worker_id, data.shape, ''.join(traceback.format_stack()))

    req = core.CreateTileReq(tile_id=tile_id, data=data)
    return self._send(tile_id, 'create', req, wait=False, timeout=timeout)

  def partial_map(self, targets, tile_ids, mapper_fn, kw, timeout=None):
    req = core.RunKernelReq(blobs=tile_ids, mapper_fn=mapper_fn, kw=kw)
    futures = self._send_all('run_kernel', req, targets=targets, timeout=timeout)
    result = {}
    for f in futures:
      for source_tile, map_result in f.iteritems():
        result[source_tile] = map_result
    return result

  def map(self, tile_ids, mapper_fn, kw, timeout=None):
    '''
    Run ``mapper_fn`` on all tiles in ``tile_ids``.
    
    Args:
      tile_ids (list): List of tiles to operate on
      mapper_fn (function): Function taking (extent, kw)
      kw (dict): Keywords to supply to ``mapper_fn``.
      timeout: optional RPC timeout.
      
    Returns:
      dict: mapping from (source_tile, result of ``mapper_fn``)
    '''
    return self.partial_map(None,
                            tile_ids,
                            mapper_fn,
                            kw,
                            timeout)

  def tile_op(self, tile_id, fn):
    '''Run ``fn`` on a single tile.
    
    Returns:
      Future: Result of ``fn``.
    '''
    req = core.TileOpReq(tile_id=tile_id, fn=fn)
    return self._send(tile_id, 'tile_op', req)
 
 
  
_ctx = threading.local()

def get():
  'Thread-local: return the context for this process.'
  return _ctx.val


def set(ctx):
  '''
  Thread-local: set the context for this process.  
  
  This is only called by the currently running worker or master.  
  '''
  _ctx.val = ctx

