#!/usr/bin/env python

'''Bootstrap code for starting a worker process.'''

from spartan import config, util
from spartan.config import flags
import atexit
import multiprocessing
import os
import resource
import socket
import spartan
import sys

try:
  import pyximport; pyximport.install()
except ImportError:
  pass


class Worker(object):
  def __init__(self):
    self.id = -1
    self._initialized = False

  def initialize(self, req, handle):
    self.id = req.id
    self._initialized = True
    handle.done()

  def create_table(self, req, handle):
    self._tables[req.id] = {}

  def destroy_table(self, req, handle):
    del self._tables[req.id]

  def get(self, req, handle):
    return self._tables[req.key]
  
  def assign_shards(self, req, handle):
    for table,shard,owner in req.assignments:
        self._tables[table].shard[shard].owner = owner

  def run_kernel(self, req, handle):
    req.kernel(self._tables[table])

  def get_iterator(self, req, handle):
    resp = IterResp()
    if req.id is None:
      table_iter = iter(self._tables[req.table])
      iter_id = id(table_iter)
      self._iters[iter_id] = table_iter
    else:
      iter_id = req.id
      table_iter = self._iters[iter_id] 
    
    resp.data = take_next(table_iter)
    resp.id = iter_id
    handle.done(resp)

  def put(self, req, handle):
    for table, shard, k, v in req.data:
      self._tables[table].shard[shard].update(k, v)

  def flush(self):
    pass

  def shutdown(self):
    pass


def _dump_kernel_prof(worker_id):
  if spartan.wrap.PROFILER is not None:
    os.system('mkdir -p ./_kernel-profiles')
    spartan.wrap.PROFILER.dump_stats('./_kernel-profiles/%d' % worker_id)

def _start_worker(master, port, local_id):
  pid = os.getpid()
  os.system('taskset -pc %d %d > /dev/null' % (local_id, pid))
  worker = spartan.start_worker(master, port)
  worker_id = worker.id()
  worker.wait_for_shutdown()
  #util.log_info('Shutting down worker... %s:%d' % (socket.gethostname(), port))
  _dump_kernel_prof(worker_id)

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
  spartan.set_log_level(flags.log_level)

  util.log_info('Starting %d workers on %s', flags.count, socket.gethostname()) 
  
  workers = []
  for i in range(flags.count):
    p = multiprocessing.Process(target=_start_worker, 
                                args=(flags.master, flags.port + i, i))
    p.start()
    workers.append(p)
    
    
  def kill_workers():
    for p in workers:
      p.terminate()
      
  watchdog = util.FileWatchdog(on_closed=kill_workers)
  watchdog.start()
  watchdog.join()
