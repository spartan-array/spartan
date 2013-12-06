#!/usr/bin/env python

import cProfile
import pstats
import threading
import time

try:
  import pyximport; pyximport.install()
except ImportError:
  pass

import multiprocessing
from multiprocessing.pool import ThreadPool
import os
import socket
import sys

import resource
from spartan import config, util, rpc, core, blob_ctx
from spartan.config import FLAGS, StrFlag, IntFlag

class Worker(object):
  def __init__(self, port, master):
    self.id = -1
    self._initialized = False
    self._peers = {}
    self._blobs = {}
    self._master = master
    self._running = True
    self._ctx = None
    self._kernel_threads = ThreadPool(processes=1)
    self._kernel_prof = cProfile.Profile()

    hostname = socket.gethostname()
    self._server = rpc.listen(hostname, port)
    self._server.register_object(self)
    self._server.serve_nonblock()

    req = core.RegisterReq()
    req.host = hostname
    req.port = port
    master.register(req)


  def get_worker(self, worker_id):
    return self._peers[worker_id]

  def initialize(self, req, handle):
    util.log_info('Worker %d initializing...', req.id)

    for id, (host, port) in req.peers.iteritems():
      self._peers[id] = rpc.connect(host, port)

    self.id = req.id
    self._ctx = blob_ctx.BlobCtx(self.id, self._peers, self)
    self._initialized = True
    handle.done()
    
  def create(self, req, handle):
    assert self._initialized
    blob_ctx.set(self._ctx)

    if req.blob_id.id == -1:
      id = blob_ctx.get().create_local()
    else:
      id = req.blob_id

    self._blobs[id] = req.data
    resp = core.CreateResp(blob_id=id)
    #util.log_info('W%d :: Created blob: id %s', self.id, id)
    handle.done(resp)

  def destroy(self, req, handle):
    for id in req.ids:
      if id in self._blobs:
        del self._blobs[id]

    #util.log_info('Destroy...')
    handle.done()

  def update(self, req, handle):
    blob =  self._blobs[req.id]
    self._blobs[req.id] = blob.update(req.data, req.reducer)

    handle.done()

  def get(self, req, handle):
    if req.selector is None:
      #util.log_info('GET: %s', type(self._blobs[req.id]))
      resp = core.GetResp(data=self._blobs[req.id])
      handle.done(resp)
    else:
      resp = core.GetResp(data=self._blobs[req.id].get(req.selector))
      handle.done(resp)

  def _run_kernel(self, req, handle):
    self._server._timers['run_kernel'].start()
    if FLAGS.profile_kernels:
      self._kernel_prof.enable()

    try:
      blob_ctx.set(self._ctx)
      results = {}
      for blob_id in req.blobs:
        #util.log_info('%s %s', blob_id, blob_id in self._blobs)
        if blob_id in self._blobs:
          #util.log_info('W%d kernel start', self.id)
          blob = self._blobs[blob_id]
          results[blob_id] = req.mapper_fn(blob_id, blob, **req.kw)
          #util.log_info('W%d kernel finish', self.id)
      handle.done(results)
    except:
      handle.exception()

    self._server._timers['run_kernel'].stop()
    self._kernel_prof.disable()

  def run_kernel(self, req, handle):
    self._kernel_threads.apply_async(self._run_kernel, args=(req, handle))

  def shutdown(self, req, handle):
    if FLAGS.profile_kernels:
      os.system('mkdir -p ./_kernel-profiles/')
      stats = pstats.Stats(self._kernel_prof)
      stats.add(rpc.poller().profiler)
      stats.dump_stats('./_kernel-profiles/%d' % self.id)

    util.log_info('Shutdown worker %d (profile? %d)', self.id, FLAGS.profile_kernels)
    #print self._server.timings()

    handle.done()
    threading.Thread(target=self._shutdown).start()

  def _shutdown(self):
    time.sleep(0.1)
    util.log_info('Closing server...')
    self._running = False
    self._server.shutdown()

  def wait_for_shutdown(self):
    while self._running:
      time.sleep(0.1)


def _start_worker(master, port, local_id):
  util.log_info('Master: %s', master)
  pid = os.getpid()
  os.system('taskset -pc %d %d > /dev/null' % (local_id, pid))
  master = rpc.connect(*master)
  worker = Worker(port, master)
  worker.wait_for_shutdown()

if __name__ == '__main__':
  sys.path.append('./tests')

  FLAGS.add(StrFlag('master'))
  FLAGS.add(IntFlag('port'))
  FLAGS.add(IntFlag('count'))

  resource.setrlimit(resource.RLIMIT_AS, (8 * 1000 * 1000 * 1000,
                                          8 * 1000 * 1000 * 1000))

  config.initialize(sys.argv)
  assert FLAGS.master is not None
  assert FLAGS.port > 0
  assert FLAGS.count > 0

  util.log_info('Starting %d workers on %s', FLAGS.count, socket.gethostname())

  m_host, m_port = FLAGS.master.split(':')
  master = (m_host, int(m_port))

  workers = []
  for i in range(FLAGS.count):
    p = multiprocessing.Process(target=_start_worker, 
                                args=(master, FLAGS.port + i, i))
    p.start()
    workers.append(p)
    
    
  def kill_workers():
    for p in workers:
      p.terminate()
      
  watchdog = util.FileWatchdog(on_closed=kill_workers)
  watchdog.start()
  watchdog.join()
