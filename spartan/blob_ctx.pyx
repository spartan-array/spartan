'''
The `BlobCtx` manages the state of a Spartan execution: it stores the
location of tiles and other workers in the system, and contains methods
for fetching and updating array data, creating and removing tiles and
running user-defined *kernel functions* on tile data.
'''


from . import util, core
import threading
from .util import Assert
import random
from rpc import serialize, Client, RemoteException, TimeoutException, WorkerProxy
from rpc import Future, FutureGroup

from libcpp.string cimport string
from libcpp.vector cimport vector
from cpython cimport bool

cdef extern from "array/ctile.h":
  cdef cppclass CTile:
    int id
    string type
    CTile()

cdef extern from "ccore.h":
  cdef cppclass TileId:
    int worker, id
    TileId()
    TileId(int, int)

  cdef cppclass Slice:
    long start, stop, step
    Slice(long, long, long)

  cdef cppclass SubSlice:
    vector[Slice] slices
    SubSlice()

  cdef cppclass EmptyMessage:
    EmptyMessage()

  cdef cppclass GetResp:
    TileId id
    string data
    GetResp()

  cdef cppclass TileIdMessage:
    TileId tile_id
    TileIdMessage()

cdef extern from "cblob_ctx.h":
  cdef cppclass CBlobCtx:
    unsigned long py_get(TileId*, SubSlice*, GetResp*) nogil
    unsigned long py_get_flatten(TileId*, SubSlice*, GetResp*) nogil
    unsigned long py_update(TileId*, SubSlice*, string*, int) nogil
    unsigned long py_create(CTile*, TileIdMessage*) nogil

ID_COUNTER = iter(xrange(10000000))

cdef class MasterBlobCtx:
  cdef dict workers
  cdef bool active

  def __init__(self, dict workers):
    '''
    Create a new context.

    Args:
      workers (list of RPC clients): RPC connections from master to workers in the computation.
        This is used to avoid sending RPC messages for operations that can be
        serviced locally.
    '''
    self.workers = workers
    self.active = True

  def is_master(self):
    '''
    True if this context is running in the master process.
    '''
    return True

  def _send(self, worker_id, method, req, wait=True, timeout=None):
    if self.active == False:
      util.log_debug('Ctx disabled.')
      return None

    fu = method(self.workers[worker_id], req)

    if wait:
      err_code = fu.wait(timeout)
      if err_code == 0:
        return fu.result
      else:
        raise RemoteException('Rpc Error: get error code %d' % err_code)
    else:
      return fu

  def _send_all(self, method, req, wait=True, timeout=None):
    if self.active == False:
      util.log_debug('Ctx disabled.')
      return None

    fu_group = FutureGroup()
    for w in self.workers.values():
      fu_group.append(method(w, req))

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

  def destroy_all(self, tile_ids):
    '''
    Destroy all tiles

    Args:
      tile_ids (list): Tiles to destroy.
    '''
    req = core.DestroyReq(ids=tile_ids)

    # Don't need to wait for the result.
    self._send_all(WorkerProxy.async_destroy, req, wait=False)

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
    req = core.GetReq(id=tile_id, subslice=subslice)

    if wait:
      return self._send(tile_id.worker, WorkerProxy.async_get, req, wait=True,
              timeout=timeout).data
    else:
      return self._send(tile_id.worker, WorkerProxy.async_get, req, wait=False)

  def get_flatten(self, tile_id, subslice, wait=True, timeout=None):
    '''
    Fetch a flatten region of the flatten format of a tile.

    Args:
      tile_id (int): Tile to fetch from.
      subslice (slice or None): Portion of tile to fetch.
      wait (boolean): Wait for this operation to finish before returning.
      timeout (float):
    '''
    req = core.GetReq(id=tile_id, subslice=subslice)

    if wait:
      return self._send(tile_id.worker, WorkerProxy.async_get_flatten, req, wait=True,
              timeout=timeout).data
    else:
      return self._send(tile_id.worker, WorkerProxy.async_get_flatten, req, wait=False)

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

    req = core.UpdateReq(id=tile_id, region=region, data=serialize(data), reducer=reducer)
    return self._send(tile_id.worker, WorkerProxy.async_update, req, wait=wait,
            timeout=timeout)

  def cancel_tile(self, worker_id, tile_id):
    '''
    Cancel the tile from the kernel remain tile list. The tile will not be executed in the specific worker.

    :param worker_id: the worker that the tile should be removed from.
    :param tile_id: tile_id of the tile to be canceled.
    '''
    req = core.TileIdMessage(tile_id=tile_id)
    return self._send(worker_id, WorkerProxy.async_cancel_tile, req)

  def create(self, data, hint=None, timeout=None):
    '''
    Create a new tile to hold ``data``.

    Args:
      data (Tile) : Tile to be created
      hint (int): Optional.  Worker to store data on.
        If not specified, workers are chosen in round-robin order.
      timeout (float):
    '''
    # master dispatches to a worker in round-robin order.
    if hint is None:
      worker_id = ID_COUNTER.next() % len(self.workers)
    else:
      worker_id = hint % len(self.workers)

    if worker_id not in self.workers:
      worker_id = random.choice(self.workers.keys())

    tile_id = core.TileId(worker=worker_id, id=-1)

    #util.log_info('%s %s %s %s', new_id, worker_id, data.shape, ''.join(traceback.format_stack()))

    req = core.CreateTileReq(tile_id=tile_id, data=data)
    return self._send(tile_id.worker, WorkerProxy.async_create, req, wait=False)

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
    req = core.RunKernelReq(blobs=tile_ids, fn=serialize((mapper_fn, kw)))
    futures = self._send_all(WorkerProxy.async_run_kernel, req)
    result = futures
    #result = {}
    #for f in futures:
        #for source_tile, map_result in f.iteritems():
      #  result[source_tile] = map_result
    return result

  def get_tile_info(self, tile_id):
    '''Get tile information on a single tile.

    Returns:
      TileInfoResp.
    '''
    req = core.TileIdMessage(tile_id=tile_id)
    return self._send(tile_id.worker, WorkerProxy.async_get_tile_info, req)

cdef class WorkerBlobCtx:
  cdef CBlobCtx* ctx

  def __cinit__(self, unsigned long ctx):
    '''
    Create a new context.

    Args:
      ctx: C++ BlobCtx used for RPC connections from worker to other workers in the computation.
        This is used to avoid sending RPC messages for operations that can be serviced locally.
    '''
    self.ctx = <CBlobCtx*>ctx

  def is_master(self):
    '''
    True if this context is running in the master process.
    '''
    return False

  def get(self, tile_id, subslice, wait=True, timeout=None):
    '''
    Fetch a region of a tile.

    Args:
      tile_id (int): Tile to fetch from.
      subslice (slice or None): Portion of tile to fetch.
      wait (boolean): Wait for this operation to finish before returning.
      timeout (float):
    '''
    cdef TileId tid
    cdef CSliceIdx s
    cdef GetResp resp

    tid.worker = tile_id.worker
    tid.id = tile_id.id
    for dim in subslice:
      s.slices.push_back(Slice(dim.start, dim.stop, dim.step))

    cdef unsigned long fu = self.ctx.py_get(&tid, &s, &resp)

    if fu == 0:
      future = Future(id=-1, rep_types=None, rep=core.GetResp(tile_id, resp.data))
    else:
      future = Future(id=fu, rep_types=['GetResp'])

    if wait:
      return future.result.data
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
    cdef TileId tid
    cdef SubSlice s
    cdef GetResp resp

    tid.worker = tile_id.worker
    tid.id = tile_id.id
    for dim in subslice:
      s.slices.push_back(Slice(dim.start, dim.stop, dim.step))

    cdef unsigned long fu = self.ctx.py_get_flatten(&tid, &s, &resp)

    if fu == 0:
      future = Future(id=-1, rep_types=None, rep=core.GetResp(tile_id, resp.data))
    else:
      future = Future(id=fu, rep_types=['GetResp'])

    if wait:
      return future.result.data
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
    cdef TileId tid
    cdef SubSlice s
    cdef string d = serialize(data)

    tid.worker = tile_id.worker
    tid.id = tile_id.id
    for dim in region:
      s.slices.push_back(Slice(dim.start, dim.stop, dim.step))

    cdef unsigned long fu = self.ctx.py_update(&tid, &s, &d, reducer)

    if fu == 0:
      future = Future(id=-1, rep_types=None, rep=core.EmptyMessage())
    else:
      future = Future(id=fu, rep_types=['EmptyMessage'])

    if wait: future.wait()
    return future

  def create(self, data, hint=None, timeout=None):
    '''
    Create a new tile to hold ``data``.

    Args:
      data (Tile): Tile to be created
      hint (int): Optional.  Worker to store data on.
        If not specified, workers are chosen in round-robin order.
      timeout (float):
    '''
    cdef CTile tile
    cdef TileIdMessage resp

    #TODO: transform the python Tile to CTile
    self.ctx.py_create(&tile, &resp)
    return Future(id=-1, rep_types=None, rep=core.TileIdMessage(core.TileId(resp.tile_id.worker, resp.tile_id.id)))

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

