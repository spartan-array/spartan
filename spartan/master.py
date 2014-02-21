'''Master process definition.

Spartan computations consist of a master and one or more workers.

The master tracks the location of array data, manages worker health,
and runs user operations on workers.
'''

import atexit
import socket
import threading
import weakref

import time
from spartan import util, rpc, core, blob_ctx
from spartan.config import FLAGS

MASTER = None

def _dump_profile():
  import yappi
  yappi.get_func_stats().save('master_prof.out', type='pstat')

def get():
  return MASTER

class Master(object):
  def __init__(self, port, num_workers):
    self._workers = {}
    self.num_workers = num_workers
    self._port = port
    self._server = rpc.listen('0.0.0.0', port)
    self._server.register_object(self)
    self._initialized = False
    self._server.serve_nonblock()
    self._ctx = None
    
    self._worker_statuses = {}
    self._worker_scores = {}
    self._worker_avg_score = 0
    self._available_workers = []
    
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

    futures = rpc.FutureGroup()
    for id, w in self._workers.iteritems():
      util.log_info('Shutting down worker %d', id)
      futures.append(w.shutdown())

    # Wait a second to let our shutdown request go out.
    time.sleep(1)

    self._server.shutdown()

  def register(self, req, handle):
    '''
    RPC method.
    
    Register a new worker with the master.
    
    Args:
      req (RegisterReq):
      handle (PendingRequest):
    '''
    id = len(self._workers)
    self._workers[id] = rpc.connect(req.host, req.port)
    self._available_workers.append(id)
    util.log_info('Registered %s:%s (%d/%d)', req.host, req.port, id, self.num_workers)

    handle.done(core.EmptyMessage())
    
    self.init_worker_score(id, req.worker_status)
    
    if len(self._workers) == self.num_workers:
      self.update_avg_score()
      threading.Thread(target=self._initialize).start()
  
  def register_array(self, array):
    self._arrays.add(array)
    
  def get_available_workers(self):
    return self._available_workers

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
    
#     completed_task_number = len(worker_status.task_reports)
#     worker_speed = 0
#     if completed_task_number > 0:  
#       for task in worker_status.task_reports:
#         worker_speed += task['finish_time'] -task['start_time']
#       worker_speed = completed_task_number / worker_speed
#     
#       self._worker_scores[worker_id] = worker_speed
#       self.update_avg_score()
  
  def update_avg_score(self):
    avg_score = 0
    for score in self._worker_scores.values():
      avg_score += score
    self._worker_avg_score = avg_score / len(self._worker_scores)
          
  def get_worker_scores(self):
    return sorted(self._worker_scores.iteritems(), key=lambda x: x[1], reverse=True)
                        
  def mark_failed_worker(self, worker_id):
    util.log_info('Marking worker %s as failed.', worker_id)
    self._available_workers.remove(worker_id)
    for array in self._arrays:
      for ex, tile_id in array.tiles.iteritems():
        if tile_id.worker == worker_id:
          array.bad_tiles.append(ex)
                                      
  def mark_failed_workers(self):
    now = time.time()
    for worker_id in self._available_workers:
      if now - self._worker_statuses[worker_id].last_report_time > FLAGS.heartbeat_interval * FLAGS.worker_failed_heartbeat_threshold:
        self.mark_failed_worker(worker_id)
  
  def is_slow_worker(self, worker_id):
    if self._worker_scores[worker_id] < self._worker_avg_score * 0.5:
      return True
    return False
  
  def heartbeat(self, req, handle):
    '''RPC method.
    
    Called by worker processes periodically.
    
    Args:
      req: `WorkerStatus`
      handle: `PendingRequest`
      
    Returns: `EmptyMessage`
    '''
    #util.log_info('Receive worker %d heartbeat.', req.worker_id)
    if req.worker_id >= 0 and self._initialized:     
      self.update_worker_score(req.worker_id, req.worker_status)
#       util.log_info('Worker scores:%s', self._worker_scores)
#       if self.is_slow_worker(req.worker_id):
#         fast_worker = max(self._worker_scores.iteritems(), key=lambda x: x[1])[0]
#         util.log_info('Slow worker: %d migrate to fast worker %d', req.worker_id, fast_worker)
      self.mark_failed_workers()
    #util.log_info('available workers:%s', self._available_workers)
    
    resp = core.EmptyMessage()
    handle.done(resp)
    #util.log_info('Finish worker %d heartbeat', req.worker_id)
      
  def _initialize(self):
    '''Sends an initialization request to all workers and waits 
    for their response.
    '''
    util.log_info('Initializing...')
    req = core.InitializeReq(peers=dict([(id, w.addr())
                                      for id, w in self._workers.iteritems()]))

    futures = rpc.FutureGroup()
    for id, w in self._workers.iteritems():
      req.id = id
      futures.append(w.initialize(req))
    futures.wait()

    self._ctx = blob_ctx.BlobCtx(blob_ctx.MASTER_ID, self._workers, self)
    self._initialized = True
    util.log_info('done...')

  def wait_for_initialization(self):
    '''Blocks until all workers are initialized.'''
    while not self._initialized:
      time.sleep(0.1)

    blob_ctx.set(self._ctx)
