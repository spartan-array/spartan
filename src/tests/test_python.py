#!/usr/bin/env python

import subprocess
import sys
import os
import cPickle

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
  sparrow.map_shards(master, table, cPickle.dumps(put_kernel))
  print table.get('123')

tests = ['test_init', 'test_put_kernel']

def main():
  import argparse
  parser = argparse.ArgumentParser()
  parser.add_argument('--run_test', default='')
  flags, argv = parser.parse_known_args()
  
  if flags.run_test:
    argv = ['python'] + argv
    master = sparrow.init(argv)
    print >>sys.stderr, 'RUNNING: ', flags.run_test
    fn = eval(flags.run_test)
    fn(master)
    return
  
  env = dict(os.environ)
  env['PYTHONPATH'] += ':src/sparrow/python:build'
  
  # setup environment, and call MPI for each test
  for t in tests:
    subprocess.Popen(['mpirun', 
                      '-x', 'PYTHONPATH', 
                      '-n', '2',
                      'python', 'src/tests/test_python.py', '--run_test=%s' % t],
                     env=env)
  
if __name__ == '__main__':
  main()
