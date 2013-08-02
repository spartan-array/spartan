#!/usr/bin/env python

import cPickle
import os
import subprocess
import sys

try:
  import sparrow
except ImportError, e:
  print 'Skipping sparrow import.', e 

def py_sharder(k, num_shards):
  return hash(k) % num_shards

def py_accum(cur, update):
  cur = update

def test_init(master):
  table = sparrow.create_table(master, py_sharder, py_accum)
  
def put_kernel():
  t = sparrow.get_table(sparrow.current_table_id())
  t.put('123', '456')

def test_put_kernel(master):
  table = sparrow.create_table(master, py_sharder, py_accum)
  sparrow.map_shards(master, table, put_kernel, tuple())
  print table.get('123')
  
  
def get_shard_kernel():
  s_id = sparrow.current_shard_id()
  t_id = sparrow.current_table_id()
  it = sparrow.get_table(t_id).get_iterator(s_id)
  
  while not it.done():
    print it.key()
    it.next()

def test_build_array(master):
  import numpy as np
  table = sparrow.create_table(master, py_sharder, py_accum)
  bytes = np.ndarray((1000, 1000), dtype=np.uint8)
  for i in range(10):
    for j in range(10):
      table.put('%d%d' % (i, j), bytes)
      table.put('%d%d' % (i, j), bytes)
      table.put('%d%d' % (i, j), bytes)
      table.put('%d%d' % (i, j), bytes)
      
  sparrow.map_shards(master, table, get_shard_kernel, tuple())

def main():
  import argparse
  parser = argparse.ArgumentParser()
  parser.add_argument('--run_test', default='')
  flags, argv = parser.parse_known_args()
  
  print 'Startup: ', flags.run_test
  
  if flags.run_test:
    import yappi
    argv = ['python'] + argv
    master = sparrow.init(argv)
    print >> sys.stderr, 'RUNNING: ', flags.run_test, ' : ', argv
    fn = eval(flags.run_test)
    fn(master)
    return
  
  env = dict(os.environ)
  env['PYTHONPATH'] += ':build:build/src/sparrow/python'
  
  # setup environment, and call MPI for each test
  tests = [k for k in globals().keys() if k.startswith('test_')]
  #tests = ['test_init']
  for t in tests:
    p = subprocess.Popen(env = env,
                         args=['mpirun', '-x', 'PYTHONPATH', '-n', '2',
                          sys.executable, 'src/tests/test_python.py', '--run_test=%s' % t])
                          #'xterm', '-e',
                          #'valgrind %s src/tests/test_python.py --run_test=%s' % (sys.executable, t)
                          #'gdb -ex run --args %s src/tests/test_python.py --run_test=%s' % (sys.executable, t)
    p.wait()
  
if __name__ == '__main__':
  main()
