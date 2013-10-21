#!/usr/bin/env python

'''Helper module for importing SWIG bindings.

Sets PYTHONPATH when running from the build directory,
and imports all symbols from the SWIG generated code.
''' 

from os.path import abspath
import sys
import collections
import logging
sys.path += [abspath('../build/.libs'), 
             abspath('../build/python/spartan/wrap'), 
             abspath('.')]

from spartan import config
from spartan_wrap import set_log_level, TableContext
import atexit
import cPickle
import cProfile
import os
import pstats
import spartan_wrap
import threading
import traceback

PYLOG_TO_CLOG = {
  logging.DEBUG : spartan_wrap.DEBUG,
  logging.INFO : spartan_wrap.INFO,
  logging.WARN : spartan_wrap.WARN,
  logging.ERROR : spartan_wrap.ERROR,
  logging.FATAL : spartan_wrap.FATAL,
}

logging.basicConfig(format='%(filename)s:%(lineno)s [%(funcName)s] %(message)s',
                    level=logging.INFO)

# log_debug = logging.debug
# log_info = logging.info
# log_warn = logging.warn
# log_error = logging.error

def findCaller(obj):
  f = sys._getframe(4)
  co = f.f_code
  filename = os.path.normcase(co.co_filename)
  return (co.co_filename, f.f_lineno, co.co_name)

root = logging.getLogger()
logging.RootLogger.findCaller = findCaller

log_mutex = threading.RLock()
def _log(msg, *args, **kw):
  level = kw.get('level', logging.INFO)
  with log_mutex:
    caller = sys._getframe(2)
    filename = caller.f_code.co_filename
    lineno = caller.f_lineno
    if 'exc_info' in kw:
      exc = ''.join(traceback.format_exc())
    else:
      exc = None
 
    if isinstance(msg, str):
      msg = msg % args
    else:
      msg = repr(msg)
   
        
    msg = str(msg)
    level = PYLOG_TO_CLOG[level]
    spartan_wrap.log(level, filename, lineno, msg)
    if exc is not None:
      spartan_wrap.log(level, filename, lineno, exc)
 
def log_info(msg, *args, **kw):
  kw['level'] = logging.INFO
  return _log(msg, *args, **kw)
  
def log_debug(msg, *args, **kw):
  kw['level'] = logging.DEBUG
  return _log(msg, *args, **kw)
  
def log_error(msg, *args, **kw):
  kw['level'] = logging.ERROR
  return _log(msg, *args, **kw)
  
def log_warn(msg, *args, **kw):
  kw['level'] = logging.WARN
  return _log(msg, *args, **kw)


class Sharder(object):
  def __call__(self, k, num_shards):
    assert False

class ModSharder(Sharder):
  def __call__(self, k, num_shards):
    return hash(k) % num_shards

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

_table_refs = collections.defaultdict(int)

class Table(spartan_wrap.Table):
  def __init__(self, id):
    _table_refs[id] += 1
    spartan_wrap.Table.__init__(self, id)
    self.thisown = False
      
  def __del__(self):
    _table_refs[self.id()] -= 1
          
  def __reduce__(self):
    return (Table, (self.id(),))
  
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
    self._kernel = spartan_wrap.cast_to_kernel(handle)
  
  def table(self, id):
    return Table(id)
  
  def args(self):
    return self._kernel.args()
  
  def current_table(self):
    return int(self.args()['table'])
  
  def current_shard(self):
    return int(self.args()['shard'])
    
PROFILER = None #cProfile.Profile()

def _with_profile(fn):
  if not config.flags.profile_kernels:
    return fn()
  
  global PROFILER
  
  if PROFILER is None:
    PROFILER = cProfile.Profile()
  
  PROFILER.enable() 
  result = fn()
  PROFILER.disable()
  return result

def _bootstrap_kernel(handle):
  kernel= Kernel(handle)
  fn, args = cPickle.loads(kernel.args()['map_args'])
  return _with_profile(lambda: fn(kernel, args))

# class BootstrapCombiner(object):
#   def __init__(self, fn):
#     self.fn = fn
#     
#   def __call__(self, *args, **kw):
#     return _with_profile(lambda: self.fn(*args, **kw))
  

class Master(object):
  def __init__(self, master, shutdown_on_del=False):
    self.shutdown_on_del = shutdown_on_del
    self._master = master
    
  def __getattr__(self, k):
    return getattr(self._master, k)
     
  def __del__(self):
    if self.shutdown_on_del:
      log_info('Shutting down master.')
      self._master.shutdown()
  
  def create_table(self, sharder, combiner, reducer, selector):
    for k, v in _table_refs.items():
      if v == 0:
        log_debug('GC, destroying table %s', k)
        self.destroy_table(k)
        del _table_refs[k]
        
    #combiner = BootstrapCombiner(combiner)
    #reducer = BootstrapCombiner(reducer)
    
    t = self._master.create_table(sharder, combiner, reducer, selector)
    return Table(t.id())
   
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
  
#   log('MAPPING: Function: %s, args: %s', fn, fn_args)
  
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
        #dst.update(-1, k, v)


def foreach_kernel(kernel, args):
  src_id, fn, kw = args
  assert not 'kernel' in kw
  kw['kernel'] = kernel
  
  src = kernel.table(src_id)
  for shard, sk, sv in src.iter(kernel.current_shard()):
#     log_info('Processing %s %s %s', shard, sk, sv)
    if has_kw_args(fn):
      fn(sk, sv, **kw)
    else:
      assert len(kw) == 1, 'Arguments passed but function does not support **kw'
      fn(sk, sv)


def map_items(table, mapper_fn, combine_fn=None, reduce_fn=None, **kw):
  master = get_master()
  
  dst = master.create_table(table.sharder(),
                            combine_fn,
                            reduce_fn,
                            table.selector())
  
  master.foreach_shard(table, mapper_kernel, 
                       (table.id(), dst.id(), mapper_fn, kw))
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
  for s, k, v in table:
    out.append((k, v))
  return out


def get_master():
  return Master(spartan_wrap.cast_to_master(spartan_wrap.TableContext.get_context()),
                shutdown_on_del = False)
  
def start_master(*args):
  m = spartan_wrap.start_master(*args)
  return Master(m, 
                shutdown_on_del=True)

def start_worker(*args):
  return spartan_wrap.start_worker(*args)

