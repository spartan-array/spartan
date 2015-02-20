'''
Functions for managing a cluster of machines.

Spartan currently supports running workers as either threads in the
current process, or by using ssh to connect to one or more
machines.

A Spartan "worker" is a single process; more than one worker can be
run on a machine; typically one worker is run per core.
'''

import os.path
import socket
import subprocess
import time
import shutil
from spartan import rpc
from spartan import util
import spartan
from spartan.config import FLAGS, AssignMode
import spartan.master


def start_remote_worker(worker, st, ed):
  '''
  Start processes on a worker machine.

  The machine will launch worker processes ``st`` through ``ed``.

  :param worker: hostname to connect to.
  :param st: First process index to start.
  :param ed: Last process to start.
  '''
  util.log_info('Starting worker %d:%d on host %s', st, ed, worker)
  if FLAGS.oprofile:
    os.system('mkdir operf.%s' % worker)

  ssh_args = ['ssh', '-oForwardX11=no', worker]

  args = ['cd %s && ' % os.path.abspath(os.path.curdir)]

  if FLAGS.xterm:
    args += ['xterm', '-e']

  if FLAGS.oprofile:
    args += ['operf -e CPU_CLK_UNHALTED:100000000', '-g', '-d', 'operf.%s' % worker]

  args += [
          #'gdb', '-ex', 'run', '--args',
          #'spartan/worker',
          os.path.dirname(spartan.__file__) + '/worker',
          '--master=%s:%d' % (socket.gethostname(), FLAGS.port_base),
          '--count=%d' % (ed - st),
          '--heartbeat_interval=%d' % FLAGS.heartbeat_interval
          ]

  # add flags from config/user
  for (name, value) in FLAGS:
    if name in ['worker_list', 'print_options']: continue
    args += [repr(value)]

  #print >>sys.stderr, args
  util.log_debug('Running worker %s', ' '.join(args))
  time.sleep(0.1)
  # TODO: improve this to make log break at newline
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
  rpc.set_default_timeout(FLAGS.default_rpc_timeout)
  #clean the checkpoint directory
  if os.path.exists(FLAGS.checkpoint_path):
    shutil.rmtree(FLAGS.checkpoint_path)

  master = spartan.master.Master(FLAGS.port_base, num_workers)

  ssh_processes = []
  if not use_cluster_workers:
    start_remote_worker('localhost', 0, num_workers)
  else:
    available_workers = sum([cnt for _, cnt in FLAGS.hosts])
    assert available_workers >= num_workers, 'Insufficient slots to run all workers.'
    count = 0
    num_hosts = len(FLAGS.hosts)
    for worker, total_tasks in FLAGS.hosts:
      if FLAGS.assign_mode == AssignMode.BY_CORE:
        sz = total_tasks
      else:
        sz = util.divup(num_workers, num_hosts)

      sz = min(sz, num_workers - count)
      ssh_processes.append(start_remote_worker(worker, count, count + sz))
      count += sz
      if count == num_workers:
        break

  master.wait_for_initialization()

  # Kill the now unnecessary ssh processes.
  # Fegin : if we kill these processes, we can't get log from workers.
  #for process in ssh_processes:
    #process.kill()
  return master
