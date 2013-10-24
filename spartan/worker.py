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
import spartan.wrap
import sys

try:
  import pyximport; pyximport.install()
except ImportError:
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