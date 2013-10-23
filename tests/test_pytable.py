#!/usr/bin/env python

from spartan import ModSharder, replace_accum
from spartan.util import Assert
from test_common import with_ctx

def fetch(table):
  out = []
  for s, k, v in table:
    out.append((k, v))
  return out

  
@with_ctx
def test_init(master):
  table = master.create_table(ModSharder(), None, replace_accum, None)

@with_ctx
def test_master(master):
  table = master.create_table(ModSharder(), None, replace_accum, None)
  table.update(0, '123', '456')
  table.flush()
  Assert.eq(table.get(0, '123'), '456')

def put_kernel(kernel, args):
  t = kernel.table(kernel.current_table())
  t.update(kernel.current_shard(), kernel.current_shard(), 1)
 
@with_ctx
def test_put_kernel(master):
  table = master.create_table(ModSharder(), None, replace_accum, None)
  master.foreach_shard(table, put_kernel, tuple())
  for i in range(table.num_shards()):
    Assert.eq(table.get(i, i), 1)

def copy_kernel(kernel, args):
  a, b = args
  ta = kernel.table(a)
  tb = kernel.table(b)
  
  for s, k, v in ta.iter(kernel.current_shard()):
    tb.update(kernel.current_shard(), k, v)
  
@with_ctx
def test_copy(master):
  src = master.create_table(ModSharder(), None, replace_accum, None)
  for i in range(100):
    src.update(i % src.num_shards(), i, i)
    
  dst = master.create_table(ModSharder(), None, replace_accum, None)
  master.foreach_shard(src, copy_kernel, (src.id(), dst.id()))
  
  src_v = fetch(src)
  dst_v = fetch(dst)
  Assert.eq(sorted(src_v), sorted(dst_v))

