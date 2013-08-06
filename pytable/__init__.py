import sys

try:
  from sparrow import *
except ImportError, e:
  print >>sys.stderr, 'Warning, sparrow library not found.', e 

def mod_sharder(k, num_shards):
  return hash(k) % num_shards

def replace_accum(cur, update):
  return update

def mapper_kernel(kernel_ptr, args):
  src_id, dst_id, fn, fn_args = args
  kernel = as_kernel(kernel_ptr)
  assert kernel.table_id() == src_id
  
  src = get_table(kernel, src_id)
  dst = get_table(kernel, dst_id)
  
  it = src.get_iterator(kernel.shard_id())
  while not it.done():
    for k, v in fn(it.key(), it.value(), *fn_args):
      dst.update(k, v)
    it.next()

def map_items(table, fn, *args):
  src = table
  dst = create_table(table.master(), mod_sharder, replace_accum)
  foreach_shard(table.master(), table, 
                mapper_kernel, (src.id(), dst.id(), fn, args))
  return dst

def map_inplace(table, fn, *args):
  src = table
  dst = src
  foreach_shard(table.master(), table, 
                mapper_kernel, (src.id(), dst.id(), fn, args))
  return dst
  

def key_mapper(k, v):
  yield k, 1
  
def keys(src):
  return map_items(src, key_mapper)