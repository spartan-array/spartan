from . import util, wrap
from spartan.config import flags
from spartan.util import Assert
from wrap import DEBUG, INFO, WARN, ERROR, FATAL, set_log_level
import cProfile
import pstats
import sys
import traceback


def mod_sharder(k, num_shards):
  return hash(k) % num_shards

def replace_accum(cur, update):
  return update

def sum_accum(cur, update):
  return cur + update

class Iter(object):
  def __init__(self, handle):
    self.handle = handle
    self._val = None
    if not wrap.iter_done(self.handle):
      self._val = (wrap.iter_key(self.handle), wrap.iter_value(self.handle)) 
    
  def __iter__(self):
    return self
    
  def next(self):
    if self._val is None:
      raise StopIteration
    
    result = self._val
    self._val = None
    wrap.iter_next(self.handle)
    if not wrap.iter_done(self.handle):
      self._val = (wrap.iter_key(self.handle), wrap.iter_value(self.handle))
    return result 

  
def key_mapper(k, v):
  yield k, 1
  
def keys(src):
  key_table = map_items(src, key_mapper)
  result = [k for k, _ in key_table]
  return result
        

class Table(object):
  def __init__(self, master, ptr_or_id):
    if master is not None:
      self.ctx = master
      self.destroy_on_del = True
    else:
      self.destroy_on_del = False
      self.ctx = wrap.get_context()
    
    if isinstance(ptr_or_id, int):
      self.handle = wrap.get_table(self.ctx, ptr_or_id)
    else:
      self.handle = ptr_or_id
      
  def __del__(self):
    if self.destroy_on_del:
      self.destroy()
          
  def __reduce__(self):
    return (Table, (None, self.id()))
    
  def id(self):
    return wrap.get_id(self.handle)
    
  def __getitem__(self, key):
    return wrap.get(self.handle, key)
  
  def __setitem__(self, key, value):
    return wrap.update(self.handle, key, value)
  
  def destroy(self):
    Assert.isinstance(self.ctx, Master) 
    return self.ctx.destroy_table(self.handle)
  
  def keys(self):
    # Don't create an iterator directly; this would have us 
    # copy all the values locally.  First construct a key-only
    # table
    return keys(self)
  
  def values(self):
    for _, v in iter(self):
      yield v
  
  def get(self, key):
    return wrap.get(self.handle, key)
  
  def update(self, key, value):
    return wrap.update(self.handle, key, value)
  
  def num_shards(self):
    return wrap.num_shards(self.handle)
  
  def flush(self):
    return wrap.flush(self.handle)
  
  def __iter__(self):
    return self.iter(-1)
  
  def iter(self, shard):
    return Iter(wrap.get_iterator(self.handle, shard))
  
  def sharder(self):
    return wrap.get_sharder(self.handle)
  
  def accum(self):
    return wrap.get_accum(self.handle)
  
  def selector(self):
    return wrap.get_selector(self.handle)
  

class Kernel(object):
  def __init__(self, kernel_id):
    self.handle = wrap.cast_to_kernel(kernel_id)
  
  def table(self, table_id):
    return Table(None, 
                 wrap.get_table(self.handle, table_id))
  
  def current_shard(self):
    return wrap.current_shard(self.handle)
  
  def current_table(self):
    return wrap.current_table(self.handle)


class Worker(object):
  def __init__(self, handle):
    self.handle = handle
    
  def wait_for_shutdown(self):
    wrap.wait_for_shutdown(self.handle)
    

PROF = None

def _bootstrap_kernel(handle, args):
  kernel = Kernel(handle)
  fn = args[0]
  rest = args[1]
  
  if not flags.profile_kernels:
    return fn(kernel, rest)
  
  p = cProfile.Profile()
  p.enable()  
  result = fn(kernel, rest)
  p.disable()
  stats = pstats.Stats(p)
  global PROF
  if PROF is None:
    PROF = stats
  else:
    PROF.add(stats)
  
  return result

class Master(object):
  def __init__(self, handle, shutdown_on_del=False):
    self.handle = handle
    self.shutdown_on_del = shutdown_on_del
     
  def __del__(self):
    if self.shutdown_on_del:
      util.log('Shutting down master.')
      wrap.shutdown(self.handle)
      
  def num_workers(self):
    return wrap.num_workers(self.handle)
    
  def destroy_table(self, table_handle):
    wrap.destroy_table(self.handle, table_handle)
    
  def create_table(self, 
                   sharder=mod_sharder, 
                   combiner=None,
                   reducer=replace_accum,
                   selector=None):
    return Table(self, 
                 wrap.create_table(self.handle, sharder, combiner, reducer, selector))
  
  def foreach_shard(self, table, kernel, args):
    return wrap.foreach_shard(
                          self.handle, table.handle, _bootstrap_kernel, (kernel, args))


def mapper_kernel(kernel, args):
  src_id, dst_id, fn = args
  
  src = kernel.table(src_id)
  dst = kernel.table(dst_id)
  
#   util.log('MAPPING: Function: %s, args: %s', fn, fn_args)
  
  for sk, sv in src.iter(kernel.current_shard()):
    result = fn(sk, sv)
    if result is not None:
      for k, v in result:
        dst.update(k, v)

def foreach_kernel(kernel, args):
  src_id, fn = args
  src = kernel.table(src_id)
  for sk, sv in src.iter(kernel.current_shard()):
    fn(sk, sv)


def map_items(table, fn):
  src = table
  master = src.ctx
  
  sharder = table.sharder()
  accum = table.accum()
  selector = table.selector()
  
  dst = master.create_table(sharder, accum, selector)
  master.foreach_shard(table, mapper_kernel, (src.id(), dst.id(), fn))
  return dst


def map_inplace(table, fn):
  src = table
  dst = src
  table.ctx.foreach_shard(table, mapper_kernel, (src.id(), dst.id(), fn))
  return dst


def foreach(table, fn):
  src = table
  master = src.ctx
  master.foreach_shard(table, foreach_kernel, (src.id(), fn))


def fetch(table):
  out = []
  for k, v in table:
    out.append((k, v))
  return out


def get_master():
  return Master(wrap.cast_to_master(wrap.get_context()),
                shutdown_on_del = False)
  
def start_master(*args):
  return Master(wrap.start_master(*args), shutdown_on_del=True)

def start_worker(*args):
  return Worker(wrap.start_worker(*args))

