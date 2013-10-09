from spartan import config, util
from spartan.config import flags
import os.path
import socket
import spartan
import subprocess
import time

def worker_loop(port): 
  watchdog = util.FileWatchdog()
  watchdog.start()
  spartan.start_worker('%s:9999' % socket.gethostname(), port)
  while 1:
    time.sleep(1)
  
def start_remote_worker(worker, st, ed):
  util.log('Starting worker %d:%d on host %s', st, ed, worker)
  args = ['ssh', 
          '-oForwardX11=no',
          worker,
          'cd %s && ' % os.path.abspath(os.path.curdir),
          #'xterm', '-e',
          #'gdb', '-ex', 'run', '--args',
          'python', '-m spartan.worker',
          '--master=%s:9999' % socket.gethostname(),
          '--count=%d' % (ed - st),
          '--port=%d' % (10000)]
  
  time.sleep(0.1)
  p = subprocess.Popen(args, executable='ssh')
  return p

def start_cluster(num_workers, local=not flags.cluster):
  master = spartan.start_master(9999, num_workers)
  spartan.set_log_level(flags.log_level)
  time.sleep(0.1)

  if local:
    for i in range(num_workers):  
      spartan.start_worker('%s:9999' % socket.gethostname(),  10000 + i)
    return master
  
  count = 0
  num_hosts = len(config.HOSTS)
  for worker, total_tasks in config.HOSTS:
    #sz = util.divup(num_workers, num_hosts)
    sz = total_tasks
    sz = min(sz, num_workers - count)
    start_remote_worker(worker, count, count + sz)
    count += sz
    if count == num_workers:
      break
    
  return master

