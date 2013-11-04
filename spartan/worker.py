#!/usr/bin/env python

import cProfile
import threading
import time


try:
  import pyximport; pyximport.install()
except ImportError:
  pass

import multiprocessing
import os
import socket
import sys

import resource
from spartan import config, util, rpc, core
from spartan.config import flags


class Worker(object):
  def __init__(self, port, master):
    self.id = -1
    self._initialized = False
    self._peers = {}
    self._blobs = {}
    self._master = master
    self._running = True
    self._ctx = None
    self._kernel_prof = cProfile.Profile()

    hostname = socket.gethostname()
    self._server = rpc.listen(hostname, port)
    self._server.register_object(self)
    self._server.serve_nonblock()

    req = core.RegisterReq()
    req.host = hostname
    req.port = port
    master.register(req).wait()


  def get_worker(self, worker_id):
    return self._peers[worker_id]

  def initialize(self, req, handle):
    util.log_info('Worker initializing...')

    for id, (host, port) in req.peers.iteritems():
      self._peers[id] = rpc.connect(host, port)

    self.id = req.id
    self._ctx = core.BlobCtx(self.id, self._peers, self)
    self._initialized = True
    handle.done()
    
  def create(self, req, handle):
    assert self._initialized
    self._blobs[req.id] = req.data

    resp = core.CreateResp(id=req.id)
    handle.done(resp)

  def destroy(self, req, handle):
    del self._blobs[req.id]
    handle.done()

  def update(self, req, handle):
    self._blobs[req.id].update(req.data, req.reducer)
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
    if flags.profile_kernels:
      self._kernel_prof.enable()

    core.set_ctx(self._ctx)
    results = {}
    for blob_id in req.blobs:
      #util.log_info('%s %s', blob_id, blob_id in self._blobs)
      if blob_id in self._blobs:
        #util.log_info('W%d kernel start', self.id)
        blob = self._blobs[blob_id]
        results[blob_id] = req.mapper_fn(blob_id, blob, **req.kw)
        #util.log_info('W%d kernel finish', self.id)
    handle.done(results)

    self._kernel_prof.disable()

  def run_kernel(self, req, handle):
    threading.Thread(target=self._run_kernel, args=(req, handle)).start()

  def shutdown(self, req, handle):
    self._running = False
    if flags.profile_kernels:
      os.system('mkdir -p ./_kernel-profiles/')
      self._kernel_prof.dump_stats('./_kernel-profiles/%d' % self.id)

    util.log_info('Shutdown worker %d', self.id)
    handle.done()
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

  util.log_info('Registering...')
  worker.wait_for_shutdown()

if __name__ == '__main__':
  sys.path.append('./tests')
  
  import spartan
  config.add_flag('master', type=str)
  config.add_flag('port', type=int)
  config.add_flag('count', type=int)
  
  resource.setrlimit(resource.RLIMIT_AS, (8 * 1000 * 1000 * 1000,
                                          8 * 1000 * 1000 * 1000))
    
  config.parse_args(sys.argv)
  assert flags.master is not None
  assert flags.port > 0
  assert flags.count > 0

  util.log_info('Starting %d workers on %s', flags.count, socket.gethostname()) 

  m_host, m_port = flags.master.split(':')
  master = (m_host, int(m_port))

  workers = []
  for i in range(flags.count):
    p = multiprocessing.Process(target=_start_worker, 
                                args=(master, flags.port + i, i))
    p.start()
    workers.append(p)
    
    
  def kill_workers():
    for p in workers:
      p.terminate()
      
  watchdog = util.FileWatchdog(on_closed=kill_workers)
  watchdog.start()
  watchdog.join()
