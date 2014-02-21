#!/usr/bin/env python

'''
This module defines the `Worker` class and related helper functions.

Workers in Spartan manage array data and computation; methods are 
available for creating, updating, reading and deleting *tiles* of
arrays.  Workers can also run a user-specified function on a set
of tiles.

Workers periodically send a heartbeat message to the master; if the
master cannot be contacted for a sufficiently long interval, workers
shut themselves down.   
'''

import cProfile
import multiprocessing
from multiprocessing.pool import ThreadPool
import os
import pstats
import socket
import sys
import threading
import time

from . import config, util, rpc, core, blob_ctx
from .config import FLAGS, StrFlag, IntFlag, BoolFlag
from .rpc import zeromq, TimeoutException
from .util import Assert
import psutil
import weakref

#timeout for hearbeat messsage
HEARTBEAT_TIMEOUT=10

class Worker(object):
  '''
  Spartan workers generally correspond to one core of a machine.
  
  Workers manage the storage of array data and running of kernel
  functions.
  
  Attributes:
      id (int): The unique identifier for this worker
      _peers (dict): Mapping from worker id to RPC client
      _blobs (dict): Mapping from tile id to tile.
  '''
  def __init__(self, master):
    self.id = -1
    self._initialized = False
    self._peers = {}
    self._blobs = {}
    self._master = master
    self._running = True
    self._ctx = None
    self.worker_status = core.WorkerStatus(psutil.TOTAL_PHYMEM, psutil.NUM_CPUS,
                                           psutil.virtual_memory().percent,
                                           psutil.cpu_percent(), 
                                           time.time(),
                                           [], [])
    
    self._lock = threading.Lock()
    
    #Patch to fix buggy assumption by multiprocessing library  
    if not hasattr(threading.current_thread(), "_children"):
      threading.current_thread()._children = weakref.WeakKeyDictionary()
    
    self._kernel_threads = ThreadPool(processes=1)
    
    if FLAGS.profile_worker:
      self._kernel_prof = cProfile.Profile()
      self._kernel_prof.disable()
      zeromq.poller().enable_profiling()
    else:
      self._kernel_prof = None

    hostname = socket.gethostname()
    self._server = rpc.listen_on_random_port(hostname)
    self._server.register_object(self)
    self._server.serve_nonblock()

    req = core.RegisterReq()
    req.host = hostname
    req.port = self._server.addr[1]
    req.worker_status = self.worker_status
    master.register(req)

  def initialize(self, req, handle):
    '''
    Initialize worker.
    
    Assigns this worker a unique identifier and sets up connections to all other workers in the process.
    
    Args:
        req (InitializeReq): foo
        handle (PendingRequest): bar
    
    '''
    util.log_debug('Worker %d initializing...', req.id)
    for id, (host, port) in req.peers.iteritems():
      self._peers[id] = rpc.connect(host, port)
    self._peers[blob_ctx.MASTER_ID] = self._master
    
    self.id = req.id
    self._ctx = blob_ctx.BlobCtx(self.id, self._peers, self)
    blob_ctx.set(self._ctx)
    self._initialized = True
    handle.done()
    
  def create(self, req, handle):
    '''
    Create a new tile.
    
    :param req: `CreateTileReq`
    :param handle: `PendingRequest`
    
    '''
    with self._lock:
      assert self._initialized
      #util.log_info('Creating: %s', req.tile_id)
      Assert.eq(req.tile_id.worker, self.id)
  
      if req.tile_id.id == -1:
        id = self._ctx.new_tile_id()
      else:
        id = req.tile_id
  
      self._blobs[id] = req.data
      resp = core.CreateTileResp(tile_id=id)
    handle.done(resp)

  def tile_op(self, req, handle):
    resp = core.RunKernelResp(result=req.fn(self._blobs[req.tile_id]))
    handle.done(resp)
        
  def destroy(self, req, handle):
    '''
    Delete zero or more blobs.
    
    :param req: `DestroyReq`
    :param handle: `PendingRequest`
    
    '''
    with self._lock:
      for id in req.ids:
        if id in self._blobs:
          del self._blobs[id]
          #util.log_info('Destroyed blob %s', id)

    #util.log_info('Destroy...')
    handle.done()

  def update(self, req, handle):
    '''
    Apply an update to a tile.
    
    :param req: `UpdateReq`
    :param handle: `PendingRequest`
    
    '''
    #util.log_info('W%d Update: %s', self.id, req.id)
    with self._lock:
      blob =  self._blobs[req.id]
      self._blobs[req.id] = blob.update(req.region, req.data, req.reducer)
    
    handle.done()

  def get(self, req, handle):
    '''
    Fetch a portion of a tile.
    
    :param req: `GetReq`
    :param handle: `PendingRequest`
    
    '''
    if req.subslice is None:
      #util.log_info('GET: %s', type(self._blobs[req.id]))
      resp = core.GetResp(data=self._blobs[req.id])
      handle.done(resp)
    else:
      resp = core.GetResp(data=self._blobs[req.id].get(req.subslice))
      handle.done(resp)

  def _run_kernel(self, req, handle):
    '''
    Run a kernel over the tiles resident on this worker.
    
    (This operation is run in a separate thread from the main worker loop).
    
    :param req: `KernelReq`
    :param handle: `PendingRequest`
    
    '''
    if FLAGS.profile_worker:
      self._kernel_prof.enable()
    
    start_time = time.time()  
    futures = []
    try:
      blob_ctx.set(self._ctx)
      results = {}
      for tile_id in req.blobs:
        #util.log_info('%s %s', tile_id, tile_id in self._blobs)
        if tile_id in self._blobs:
          #util.log_info('W%d kernel start', self.id)
          blob = self._blobs[tile_id]
          map_result = req.mapper_fn(tile_id, blob, **req.kw)
          results[tile_id] = map_result.result
          
          if map_result.futures is not None:
            futures.append(map_result.futures)

          
          #util.log_info('W%d kernel finish', self.id)
      
      # wait for all kernel update operations to finish
#       util.log_warn('Waiting for %s futures', len(futures))
      rpc.wait_for_all(futures) 
      handle.done(results)
    except:
      util.log_warn('Exception occurred during kernel call', exc_info=1)
      self.worker_status.add_task_failure(req)
      handle.exception()
    
    finish_time = time.time()
    self.worker_status.add_task_report(req, start_time, finish_time)
    
    if FLAGS.profile_worker:
      self._kernel_prof.disable()

  def run_kernel(self, req, handle):
    '''
    Run a kernel on tiles local to this worker.
    
    :param req: `KernelReq`
    :param handle: `PendingRequest`
    
    '''
    self._kernel_threads.apply_async(self._run_kernel, args=(req, handle))
      
  def shutdown(self, req, handle):
    '''
    Shutdown this worker.
    
    Shutdown is deferred to another thread to ensure the RPC reply
    is sent before the poll loop is killed.
    
    :param req: `EmptyMessage`
    :param handle: `PendingRequest`
    
    '''
    util.log_info('Shutdown worker %d (profile? %d)', self.id, FLAGS.profile_worker)
    if FLAGS.profile_worker:
      try:
        os.system('mkdir -p ./_worker_profiles/')
        stats = pstats.Stats(zeromq.poller().profiler, self._kernel_prof)
        stats.dump_stats('./_worker_profiles/%d' % self.id)
      except Exception, ex:
        print 'Failed to write profile.', ex
    handle.done()
    threading.Thread(target=self._shutdown).start()

  def _shutdown(self):
    util.log_debug('Closing server...')
    time.sleep(0.1)
    self._running = False
    self._server.shutdown()
  
  def wait_for_shutdown(self):
    '''
    Wait for the worker to shutdown.
    
    Periodically send heartbeat updates to the master.
    '''
    last_heartbeat = time.time()
    while self._running:
      now = time.time()
      if now - last_heartbeat < FLAGS.heartbeat_interval or not self._initialized:
        time.sleep(0.1)
        continue
      
      self.worker_status.update_status(psutil.virtual_memory().percent, psutil.cpu_percent(), now)
      future = self._ctx.heartbeat(self.worker_status, HEARTBEAT_TIMEOUT)  
      try:
        future.wait()
      except TimeoutException:
        util.log_info("Exit due to heartbeat message timeout.")
        sys.exit(0)

      last_heartbeat = time.time()


def _start_worker(master, local_id):
  '''
  Start a worker, register it with the master process, and wait
  until the worker is shutdown.
  
  Runs in a subprocess.
  
  :param master: (host, port)
  :param local_id: index (from 0..#num_workers) of this worker on the local machine.
  '''
  util.log_info('Worker starting up... Master: %s Profile: %s', master, FLAGS.profile_worker)
  rpc.set_default_timeout(FLAGS.default_rpc_timeout)
  if FLAGS.use_single_core:
    pid = os.getpid()
    os.system('taskset -pc %d %d > /dev/null' % (local_id * 2, pid))
    
  master = rpc.connect(*master)
  worker = Worker(master)
  worker.wait_for_shutdown()

if __name__ == '__main__':
  sys.path.append('./tests')

  FLAGS.add(StrFlag('master'))
  FLAGS.add(IntFlag('count'))
  FLAGS.add(IntFlag('heartbeat_interval'))

  #resource.setrlimit(resource.RLIMIT_AS, (8 * 1000 * 1000 * 1000,
  #                                        8 * 1000 * 1000 * 1000))

  config.parse(sys.argv)
  assert FLAGS.master is not None
  assert FLAGS.count > 0
  assert FLAGS.heartbeat_interval > 0

  util.log_info('Starting %d workers on %s', FLAGS.count, socket.gethostname())

  m_host, m_port = FLAGS.master.split(':')
  master = (m_host, int(m_port))

  workers = []
  for i in range(FLAGS.count):
    p = multiprocessing.Process(target=_start_worker, 
                                args=(master, i))
    p.start()
    workers.append(p)
    
    
  def kill_workers():
    for p in workers:
      p.terminate()
      
  watchdog = util.FileWatchdog(on_closed=kill_workers)
  watchdog.start()
  
  for w in workers:
    w.join()
    
  print >>sys.stderr, 'All workers exited!'
