from spartan import config, util
from spartan.config import flags
import os.path
import socket
import spartan
import spartan.worker
import subprocess
import threading
import time

def start_remote_worker(worker, st, ed):
  util.log_info('Starting worker %d:%d on host %s', st, ed, worker)
  if flags.use_threads and worker == 'localhost':
    for i in range(st, ed):
      p = threading.Thread(target=spartan.worker._start_worker,
                           args=('%s:%d' % (socket.gethostname(), flags.port_base), 
                                 flags.port_base + 1 + i,
                                 i))
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

def start_cluster(num_workers, cluster):
  master = spartan.start_master(flags.port_base, num_workers)
  spartan.set_log_level(flags.log_level)
  time.sleep(0.1)

  if not cluster:
    start_remote_worker('localhost', 0, num_workers)
    return master
  
  count = 0
  num_hosts = len(config.HOSTS)
  for worker, total_tasks in config.HOSTS:
    if flags.assign_mode == config.AssignMode.BY_CORE:
      sz = total_tasks
    else:
      sz = util.divup(num_workers, num_hosts)
    
    sz = min(sz, num_workers - count)
    start_remote_worker(worker, count, count + sz)
    count += sz
    if count == num_workers:
      break
    
  return master

