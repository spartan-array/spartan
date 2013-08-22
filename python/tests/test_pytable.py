#!/usr/bin/env python

from spartan import mod_sharder, replace_accum, fetch
from spartan.util import Assert
import sys
import test_common
  
def test_init(master):
  table = master.create_table(mod_sharder, replace_accum)

def test_master(master):
  table = master.create_table(mod_sharder, replace_accum)
  table.update('123', '456')
  Assert.eq(table.get('123'), '456')

def put_kernel(kernel, args):
  t = kernel.table(kernel.current_table())
  t.update(kernel.current_shard(), 1)
 
def test_put_kernel(master):
  table = master.create_table(mod_sharder, replace_accum)
  master.foreach_shard(table, put_kernel, tuple())
  for i in range(table.num_shards()):
    Assert.eq(table.get(i), 1)

def copy_kernel(kernel, args):
  a, b = args
  ta = kernel.table(a)
  tb = kernel.table(b)
  
  for k, v in ta.iter(kernel.current_shard()):
    tb.update(k, v)
  
def test_copy(master):
  src = master.create_table(mod_sharder, replace_accum)
  for i in range(100):
    src.update(i, i)
    
  dst = master.create_table(mod_sharder, replace_accum)
  master.foreach_shard(src, copy_kernel, (src.id(), dst.id()))
  
  src_v = fetch(src)
  dst_v = fetch(dst)
  Assert.eq(sorted(src_v), sorted(dst_v))

if __name__ == '__main__':
  test_common.run_cluster_tests(sys.modules['__main__'].__file__)
