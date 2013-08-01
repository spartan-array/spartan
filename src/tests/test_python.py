#!/usr/bin/env python

import subprocess
import sys
import os
import cPickle
import numpy as np

try:
  import sparrow
except ImportError:
  print 'Skipping sparrow import.'

def test_init(master):
  table = master.create_table()
  
def put_kernel():
  t = sparrow.get_table(sparrow.current_table_id())
  t.put('123', '456')

def test_put_kernel(master):
  table = master.create_table()
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
  table = master.create_table()
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
  for t in tests:
    p = subprocess.Popen(['mpirun',
                          '-x', 'PYTHONPATH',
                          '-n', '2',
                          'python', 'src/tests/test_python.py', 
                          '--run_test=%s' % t,
                          ] + argv, env=env)
    p.wait()
  
if __name__ == '__main__':
  main()
