from . import util, wrap
from spartan.config import flags
from spartan.util import Assert
from wrap import DEBUG, INFO, WARN, ERROR, FATAL, set_log_level
import cPickle
import cProfile
import pstats
import sys
import traceback

class Sharder(object):
  def __call__(self, k, num_shards):
    assert False

class ModSharder(Sharder):
  def __call__(self, k, num_shards):
    return k.shard() % num_shards

def replace_accum(key, cur, update):
  return update

def sum_accum(key, cur, update):
  return cur + update

class Iter(object):
  def __init__(self, table, shard):
    self._table = table
    if shard == -1:
      wrap_iter = self._table.get_iterator()
    else:
      wrap_iter = self._table.get_iterator(shard)
      
    self._iter = wrap_iter
    self._val = None
    
    if not self._iter.done():
      self._val = (self._iter.shard(), self._iter.key(), self._iter.value())
    
  def __iter__(self):
    return self
  
  def next(self):
    if self._val is None:
      raise StopIteration
    
    result = self._val
    self._val = None
    
    self._iter.next()
    if not self._iter.done():
      self._val = (self._iter.shard(), self._iter.key(), self._iter.value())
    
    return result 

  
def key_mapper(k, v):
  yield k, 1
  
def keys(src):
  return map_items(src, key_mapper)

class Table(wrap.PyTable):
  def __init__(self, id, destroy_on_del=False):
    #print 'Creating table: %d, destroy? %d' % (id, destroy_on_del)
    wrap.PyTable.__init__(self, id)
    self.thisown = False
    self.destroy_on_del = destroy_on_del
      
  def __del__(self):
    if self.destroy_on_del:
      get_master().destroy_table(self)
          
  def __reduce__(self):
    return (Table, (self.id(), False))
  
  def iter(self, shard=-1):
    return Iter(self, shard)
  
  def __iter__(self):
    return self.iter()
    
  def keys(self):
    # Don't create an iterator directly; this would have us 
    # copy all the values locally.  Instead construct a key-only table
    return keys(self)
  
  def values(self):
    for _, v in iter(self):
      yield v

class Kernel(object):
  def __init__(self, handle):
    self._kernel = wrap.cast_to_kernel(handle)
  
  def table(self, id):
    return Table(id)
  
  def args(self):
    return self._kernel.args()
  
  def current_shard(self):
    return int(self.args()['shard'])
    
PROF = None

def _bootstrap_kernel(handle):
  kernel= Kernel(handle)
  fn, args = cPickle.loads(kernel.args()['map_args'])
 
  if not flags.profile_kernels:
    return fn(kernel, args)
  
  p = cProfile.Profile()
  p.enable()  
  result = fn(kernel, args)
  p.disable()
  stats = pstats.Stats(p)
  global PROF
  if PROF is None:
    PROF = stats
  else:
    PROF.add(stats)
  
  return result

class Master(object):
  def __init__(self, master, shutdown_on_del=False):
    self.shutdown_on_del = shutdown_on_del
    self._master = master
    
  def __getattr__(self, k):
    return getattr(self._master, k)
     
  def __del__(self):
    if self.shutdown_on_del:
      util.log('Shutting down master.')
      self._master.shutdown()
  
  def create_table(self, *args):
    t = self._master.create_table(*args)
    return Table(t.id(), destroy_on_del=True)
   
  def foreach_shard(self, table, kernel, args):
    return self._master.foreach_shard(table, _bootstrap_kernel, (kernel, args))

  def foreach_worklist(self, worklist, mapper):
    mod_wl = []
    for args, locality in worklist:
      mod_wl.append(((mapper, args), locality))
      
    return self._master.foreach_worklist(mod_wl, _bootstrap_kernel)


def has_kw_args(fn):
  return fn.__code__.co_flags & 0x08

def mapper_kernel(kernel, args):
  src_id, dst_id, fn, kw = args
  
  src = kernel.table(src_id)
  dst = kernel.table(dst_id)
  
  assert not 'kernel' in kw
  kw['kernel'] = kernel
  
#   util.log('MAPPING: Function: %s, args: %s', fn, fn_args)
  
  shard = kernel.current_shard()
  
  for _, sk, sv in src.iter(kernel.current_shard()):
    if has_kw_args(fn):
      result = fn(sk, sv, **kw)
    else:
      assert len(kw) == 1, 'Arguments passed but function does not support **kw'
      result = fn(sk, sv)
      
    if result is not None:
      for k, v in result:
        dst.update(shard, k, v)


def foreach_kernel(kernel, args):
  src_id, fn, kw = args
  assert not 'kernel' in kw
  kw['kernel'] = kernel
  
  src = kernel.table(src_id)
  for _, sk, sv in src.iter(kernel.current_shard()):
    if has_kw_args(fn):
      fn(sk, sv, **kw)
    else:
      assert len(kw) == 1, 'Arguments passed but function does not support **kw'
      fn(sk, sv)


def map_items(table, fn, **kw):
  src = table
  master = get_master()
  
  dst = master.create_table(table.sharder(), 
                            table.combiner(), 
                            table.reducer(),
                            table.selector())
  master.foreach_shard(table, mapper_kernel, 
                       (src.id(), dst.id(), fn, kw))
  return dst


def map_inplace(table, fn, **kw):
  src = table
  dst = src
  master = get_master()
  master.foreach_shard(table, mapper_kernel, 
                          (src.id(), dst.id(), fn, kw))
  return dst


def foreach(table, fn, **kw):
  src = table
  master = get_master()
  master.foreach_shard(table, foreach_kernel, 
                       (src.id(), fn, kw))


def fetch(table):
  out = []
  for k, v in table:
    out.append((k, v))
  return out


def get_master():
  return Master(wrap.cast_to_master(wrap.TableContext.get_context()),
                shutdown_on_del = False)
  
def start_master(*args):
  m = wrap.start_master(*args)
  return Master(m, 
                shutdown_on_del=True)

def start_worker(*args):
  return wrap.start_worker(*args)

