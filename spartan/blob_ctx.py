'''
The `BlobCtx` manages the state of a Spartan execution: it stores the
location of tiles and other workers in the system, and contains methods
for fetching and updating array data, creating and removing tiles and
running user-defined *kernel functions* on tile data.
'''


from . import util, core, _cblob_ctx_py_if
import threading
import random
import numpy as np
from rpc import serialize, RemoteException, WorkerProxy
from rpc import FutureGroup, Future
import rpc.rpc_array
from .array.tile import builtin_reducers


ID_COUNTER = iter(xrange(10000000))
MASTER_ID = 65536


class BlobCtx(object):
  def __init__(self, worker_id, workers, clients, cblob_ctx=0):
    '''
    Create a new context.

    Args:
      workers (list of RPC clients): RPC connections from master to workers in
      the computation. This is used to avoid sending RPC messages for operations
      that can be serviced locally.
    '''
    self.workers = workers
    self.worker_id = worker_id
    self.active = True

    cclients = {}
    if (self.is_master()):
      assert cblob_ctx == 0
      self.num_workers = len(workers)
      for k, v in clients.iteritems():
        cclients[k] = v.id

    self._cblob_ctx = _cblob_ctx_py_if.CBlobCtx_Py(worker_id, cclients, cblob_ctx)

  def is_master(self):
    '''
    True if this context is running in the master process.
    '''
    return self.worker_id == MASTER_ID

  def _send(self, worker_id, method, req, wait=True, timeout=None):
    if not self.active:
      util.log_debug('Ctx disabled.')
      return None

    fu = method(self.workers[worker_id], req)
    # Transform simplerpc.future to our future.
    fu = Future(fu=fu)
    assert isinstance(fu, Future), type(fu)

    if wait:
      err_code = fu.wait(timeout)
      if err_code == 0:
        return fu.result
      else:
        raise RemoteException('Rpc Error: get error code %d' % err_code)
    else:
      return fu

  def _send_all(self, method, req, wait=True, timeout=None):
    if not self.active:
      util.log_debug('Ctx disabled.')
      return None

    fu_group = FutureGroup()
    for w in self.workers.values():
      fu = method(w, req)
      # Transform simplerpc.future to our future.
      fu = Future(fu=fu)
      fu_group.append(fu)

    if wait:
      result = []
      for fu in fu_group:
        err_code = fu.wait(timeout)
        if err_code == 0:
          result.append(fu.result)
        else:
          raise RemoteException('Rpc Error: get error code %d' % err_code)
      return result
    else:
      return fu_group

  def create(self, data, hint=None, timeout=None):
    '''
    Create a new tile to hold ``data``.

    Args:
      data (Tile) : Tile to be created
      hint (int): Optional.  Worker to store data on.
        If not specified, workers are chosen in round-robin order.
      timeout (float):
    '''
    assert not isinstance(data, np.ndarray)
    # master dispatches to a worker in round-robin order.
    if self.is_master():
      if hint is None:
        worker_id = ID_COUNTER.next() % len(self.workers)
      else:
        worker_id = hint % len(self.workers)

      if worker_id not in self.workers:
        worker_id = random.choice(self.workers.keys())
    else:
      worker_id = self.worker_id
    tile_id = core.TileId(worker=worker_id, id=-1)

    return self._cblob_ctx.create(tile_id, data)

  def get(self, tile_id, subslice, wait=True, timeout=None):
    '''
    Fetch a region of a tile.

    Args:
      tile_id (int): Tile to fetch from.
      subslice (slice or None): Portion of tile to fetch.
      wait (boolean): Wait for this operation to finish before returning.
      timeout (float):
    '''
    future = self._cblob_ctx.get(tile_id, subslice)

    if wait:
      return future.result
    else:
      return future

  def get_flatten(self, tile_id, subslice, wait=True, timeout=None):
    '''
    Fetch a flatten region of the flatten format of a tile.

    Args:
      tile_id (int): Tile to fetch from.
      subslice (slice or None): Portion of tile to fetch.
      wait (boolean): Wait for this operation to finish before returning.
      timeout (float):
    '''
    future = self._cblob_ctx.get(tile_id, subslice)

    if wait:
      return future.result
    else:
      return future

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
    ctile = rpc.rpc_array.numpy_to_ctile(data)
    if builtin_reducers.get(reducer, None) is None:
      _reducer = reducer
    else:
      _reducer = builtin_reducers[reducer]
    future = self._cblob_ctx.update(tile_id, region, ctile, _reducer)
    #rpc_array.release_ctile(ctile)

    if wait:
      return future.result
    else:
      return future

  def destroy_all(self, tile_ids):
    '''
    Destroy all tiles

    Args:
      tile_ids (list): Tiles to destroy.
    '''
    assert self.worker_id == MASTER_ID
    req = core.DestroyReq(ids=tile_ids)

    # Don't need to wait for the result.
    self._send_all(WorkerProxy.async_destroy, req, wait=False)

  def destroy(self, tile_id):
    '''
    Destroy a tile.

    Args:
      tile_id (int): Tile to destroy.
    '''
    assert self.worker_id == MASTER_ID
    return self.destroy_all([tile_id])

  def cancel_tile(self, worker_id, tile_id):
    '''
    Cancel the tile from the kernel remain tile list.
    The tile will not be executed in the specific worker.

    :param worker_id: the worker that the tile should be removed from.
    :param tile_id: tile_id of the tile to be canceled.
    '''
    assert self.worker_id == MASTER_ID
    req = core.TileIdMessage(tile_id=tile_id)
    return self._send(worker_id, WorkerProxy.async_cancel_tile, req)

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
    assert self.worker_id == MASTER_ID
    req = core.RunKernelReq(blobs=tile_ids, fn=serialize((mapper_fn, kw)))
    futures = self._send_all(WorkerProxy.async_run_kernel, req)
    result = {}
    for f in futures:
      for source_tile, map_result in f.result.iteritems():
        result[source_tile] = map_result
    return result

  def get_tile_info(self, tile_id):
    '''Get tile information on a single tile.

    Returns:
      TileInfoResp.
    '''
    req = core.TileIdMessage(tile_id=tile_id)
    return self._send(tile_id.worker, WorkerProxy.async_get_tile_info, req)

_ctx = threading.local()
_ctx.val = None


def get():
  'Thread-local: return the context for this process.'
  return _ctx.val


def set(ctx):
  '''
  Thread-local: set the context for this process.

  This is only called by the currently running worker or master.
  '''
  _ctx.val = ctx

