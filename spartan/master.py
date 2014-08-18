'''Master process definition.

Spartan computations consist of a master and one or more workers.

The master tracks the location of array data, manages worker health,
and runs user operations on workers.
'''
import atexit
import threading
import weakref
import time
import spartan
from spartan import util, core, blob_ctx
from spartan.config import parse, FLAGS
from spartan.fastrpc import MasterService, WorkerProxy, Future, FutureGroup, Server, Client

MASTER = None
def _dump_profile():
  import yappi
  yappi.get_func_stats().save('master_prof.out', type='pstat')

def get():
  return MASTER

class Master(MasterService):
  def __init__(self, port, num_workers):
    self._workers = {}
    self.num_workers = num_workers
    self._server = Server(4)
    self._server.reg_svc(self)
    self._server.start("0.0.0.0:%d" % port)
    self._initialized = False
    self._ctx = None
    
    self._worker_statuses = {}
    self._worker_scores = {}
    self._available_workers = {}
    
    self._arrays = weakref.WeakSet()
    
    if FLAGS.profile_master:
      import yappi
      yappi.start()
      atexit.register(_dump_profile)

    global MASTER
    MASTER = self
    
  def __del__(self):
    # Make sure that we shutdown the cluster when the master goes away.
    self.shutdown()

  def shutdown(self):
    '''Shutdown all workers and halt.'''
    if self._ctx.active is False:
      return

    self._ctx.active = False

    futures = FutureGroup()
    req = core.EmptyMessage()
    for id, w in self._available_workers.iteritems():
      util.log_info('Shutting down worker %d', id)
      futures.append(w.async_shutdown(req))

    # Wait a second to let our shutdown request go out.
    time.sleep(1)

  def reg(self, req):
    '''
    RPC method.
    
    Register a new worker with the master.
    
    Args:
      req (RegisterReq):
      handle (PendingRequest):
    '''
    id = len(self._workers)
    c = Client()
    c.connect(req.host)
    self._workers[id] = req.host
    self._available_workers[id] = WorkerProxy(c)
    util.log_info('Registered %s (%d/%d)', req.host, id, self.num_workers)
    
    self.init_worker_score(id, req.worker_status)
    
    if len(self._workers) == self.num_workers:
      threading.Thread(target=self._initialize).start()
    return

  def register_array(self, array):
    self._arrays.add(array)
    
  def get_available_workers(self):
    return self._available_workers.keys()

  def get_workers_for_reload(self, array):
    tile_in_worker = [[i, 0] for i in range(self.num_workers)]
    for tile_id in array.tiles.values():
      tile_in_worker[tile_id.worker][1] += 1
    
    tile_in_worker = [tile_in_worker[worker_id] for worker_id in self._available_workers]
    
    tile_in_worker.sort(key=lambda x : x[1])
    result = {}
    for i in range(len(array.bad_tiles)):
      result[array.bad_tiles[i]] = tile_in_worker[i%len(tile_in_worker)][0]
    
    return result
      
  def init_worker_score(self, worker_id, worker_status):
    self._worker_statuses[worker_id] = worker_status
    self._worker_scores[worker_id] = (100 - worker_status.mem_usage) * worker_status.total_physical_memory / 1e13 #0.1-0.3
    
  def update_worker_score(self, worker_id, worker_status):
    self._worker_statuses[worker_id] = worker_status
          
  def get_worker_scores(self):
    return sorted(self._worker_scores.iteritems(), key=lambda x: x[1], reverse=True)
                        
  def mark_failed_worker(self, worker_id):
    util.log_info('Marking worker %s as failed.', worker_id)
    del self._available_workers[worker_id]
    for array in self._arrays:
      for ex, tile_id in array.tiles.iteritems():
        if tile_id.worker == worker_id:
          array.bad_tiles.append(ex)
                                      
  def mark_failed_workers(self):
    now = time.time()
    for worker_id in self._available_workers:
      if now - self._worker_statuses[worker_id].last_report_time > FLAGS.heartbeat_interval * FLAGS.worker_failed_heartbeat_threshold:
        self.mark_failed_worker(worker_id)
       
  def maybe_steal_tile(self, req):
    '''
    RPC Method
    This is called when a worker has finished processing all of it's current tiles,
    and is looking for more work to do. 
    We check if there are any outstanding tiles on existing workers to steal from.
    
    Args:
      req (UpdateAndStealTileReq):
      handle (PendingRequest):
    '''
    self._worker_statuses[req.worker_id].kernel_remain_tiles = []
    
    # update the migrated tile
    if req.old_tile_id is not None:
      util.log_debug('worker(%s) update old_tile:%s new_tile:%s', req.worker_id, req.old_tile_id, req.new_tile_id)
      for array in self._arrays:
        ex = array.blob_to_ex.get(req.old_tile_id)
        if ex is not None:
          array.tiles[ex] = req.new_tile_id
          array.blob_to_ex[req.new_tile_id] = ex
          
          del array.blob_to_ex[req.old_tile_id]
          self._ctx.destroy(req.old_tile_id)
          break
    
    # apply a new tile for execution    
    slow_workers = sorted(self._worker_statuses.iteritems(), 
                          key=lambda x: len(x[1].kernel_remain_tiles), 
                          reverse=True)
    for slow_worker in slow_workers:
      if len(slow_worker[1].kernel_remain_tiles) == 0: break
        
      tile_id = slow_worker[1].kernel_remain_tiles[0]
      if self._ctx.cancel_tile(slow_worker[0], tile_id):
        util.log_warn('move tile:%s from worker(%s) to worker(%s)', tile_id, slow_worker[0], req.worker_id)
        slow_worker[1].kernel_remain_tiles.remove(tile_id)
        return core.TileIdMessage(tile_id)
 
    return core.TileIdMessage(core.TileId(-1,-1))
    
  def heartbeat(self, req):
    '''RPC method.
    
    Called by worker processes periodically.
    
    Args:
      req: `WorkerStatus`
      handle: `PendingRequest`
      
    Returns: `EmptyMessage`
    '''
    #util.log_info('Receive worker %d heartbeat:%s', req.worker_id, req.worker_status)
    if req.worker_id >= 0 and self._initialized:     
      self.update_worker_score(req.worker_id, req.worker_status)
      self.mark_failed_workers()
      #util.log_info('available workers:%s', self._available_workers)
    
    return core.EmptyMessage()
      
  def _initialize(self):
    '''Sends an initialization request to all workers and waits 
    for their response.
    '''
    util.log_info('Initializing...')
    req = core.InitializeReq(0, self._workers)

    futures = FutureGroup()
    for id, w in self._available_workers.iteritems():
      req = req._replace(id=id)
      futures.append(w.async_initialize(req))
    futures.wait()

    self._ctx = blob_ctx.MasterBlobCtx(self._available_workers)
    self._initialized = True
    util.log_info('done...')

  def wait_for_initialization(self):
    '''Blocks until all workers are initialized.'''
    while not self._initialized:
      time.sleep(0.1)

    blob_ctx.set(self._ctx)

