import os.path
import socket
import subprocess
import threading
import time

from spartan import config, util
from spartan.config import flags
import spartan
import spartan.worker
import spartan.master

def _start_remote_worker(worker, st, ed):
  util.log_info('Starting worker %d:%d on host %s', st, ed, worker)
  if flags.use_threads and worker == 'localhost':
    util.log_info('Using threads.')
    for i in range(st, ed):
      p = threading.Thread(target=spartan.worker._start_worker,
                           args=((socket.gethostname(), flags.port_base), flags.port_base + 1 + i, i))
      p.daemon = True
      p.start()
    time.sleep(0.1)
    return
  
  if flags.oprofile:
    os.system('mkdir operf.%s' % worker)
    
  ssh_args = ['ssh', '-oForwardX11=no', worker ]
 
  args = ['cd %s && ' % os.path.abspath(os.path.curdir)]
   
  if flags.oprofile:
    args += ['operf -e CPU_CLK_UNHALTED:100000000', '-g', '-d', 'operf.%s' % worker]
  
  args += [          
          #'xterm', '-e',
          #'gdb', '-ex', 'run', '--args',
          'python', '-m spartan.worker',
          '--master=%s:%d' % (socket.gethostname(), flags.port_base),
          '--count=%d' % (ed - st),
          '--port=%d' % (flags.port_base + 1)]
  
  for name, value in config.flags:
    if isinstance(value, bool):
      value = int(value)
    args.append('--%s=%s' % (name, value))
  
  #print args
  time.sleep(0.1)

  if worker != 'localhost':
    p = subprocess.Popen(ssh_args + args, executable='ssh')
  else:
    p = subprocess.Popen(' '.join(args), shell=True, stdin=subprocess.PIPE)
    
  return p

def start_cluster(num_workers, use_cluster_workers):
  '''
  Start a cluster with ``num_workers`` workers.
  
  If use_cluster_workers is True, then use the remote workers
  defined in `spartan.config`.  Otherwise, workers are all
  spawned on the localhost.
  
  :param num_workers:
  :param use_cluster_workers:
  '''
  if not use_cluster_workers:
    _start_remote_worker('localhost', 0, num_workers)
  else:
    count = 0
    num_hosts = len(config.HOSTS)
    for worker, total_tasks in config.HOSTS:
      if flags.assign_mode == config.AssignMode.BY_CORE:
        sz = total_tasks
      else:
        sz = util.divup(num_workers, num_hosts)
      
      sz = min(sz, num_workers - count)
      _start_remote_worker(worker, count, count + sz)
      count += sz
      if count == num_workers:
        break

  master = spartan.master.Master(flags.port_base, num_workers)
  time.sleep(0.1)
  master.wait_for_initialization()
  return master

