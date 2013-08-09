import traceback

try:
  import sparrow
except ImportError, e:
  print 'Native module import failed:', e

class Iter(object):
  def __init__(self, handle):
    self.handle = handle
    self._val = None
    if not sparrow.iter_done(self.handle):
      self._val = (sparrow.iter_key(self.handle), sparrow.iter_value(self.handle)) 
    
  def __iter__(self):
    return self
    
  def next(self):
    if self._val is None:
      raise StopIteration
    
    result = self._val
    self._val = None
    sparrow.iter_next(self.handle)
    if not sparrow.iter_done(self.handle):
      self._val = (sparrow.iter_key(self.handle), sparrow.iter_value(self.handle))
    return result 
  

class Table(object):
  def __init__(self, master, handle, sharder, accum, selector):
    self.master = master
    self.handle = handle
    self.sharder = sharder
    self.accum = accum
    self.selector = selector
    
  def id(self):
    return sparrow.get_id(self.handle)
    
  def __getitem__(self, key):
    return sparrow.get(self.handle, key)
  
  def __setitem__(self, key, value):
    return sparrow.update(self.handle, key, value)
  
  def get(self, key):
    return sparrow.get(self.handle, key)
  
  def update(self, key, value):
    return sparrow.update(self.handle, key, value)
  
  def __iter__(self):
    return self.iter(-1)
  
  def iter(self, shard):
    return Iter(sparrow.get_iterator(self.handle, shard))
  

class Kernel(object):
  def __init__(self, handle):
    self.handle = handle
  
  def table(self, table_id):
    return Table(None, sparrow.get_table(self.handle, table_id), 
                 None, None, None)
  
  def current_shard(self):
    return sparrow.current_shard(self.handle)
  
  def current_table(self):
    return sparrow.current_table(self.handle)


def _bootstrap_kernel(handle, args):
  kernel = Kernel(handle)
  fn = args[0]
  rest = args[1]
  return fn(kernel, rest)

class Master(object):
  def __init__(self, handle):
    self.handle = handle
    
  def __del__(self):
    #print 'Shutting down!'
    #traceback.print_stack()
    sparrow.shutdown(self.handle)
    
  def create_table(self, sharder, accum, selector=None):
    return Table(self, sparrow.create_table(self.handle, sharder, accum, selector),
                 sharder, accum, selector)
  
  def foreach_shard(self, table, kernel, args):
    return sparrow.foreach_shard(self.handle, table.handle, _bootstrap_kernel, 
                                 (kernel, args))


def init(argv):
  return Master(sparrow.init(argv))


def mod_sharder(k, num_shards):
  return hash(k) % num_shards

def replace_accum(cur, update):
  return update

def sum_accum(cur, update):
  return cur + update


def mapper_kernel(kernel, args):
  src_id, dst_id, fn, fn_args = args
  
  src = kernel.table(src_id)
  dst = kernel.table(dst_id)
  
  for sk, sv in src.iter(kernel.current_shard()):
    for k, v in fn(sk, sv, *fn_args):
      dst.update(k, v)


def map_items(table, fn, *args):
  src = table
  master = src.master
  
  sharder = table.sharder
  accum = table.accum
  selector = table.selector
  
  dst = master.create_table(sharder, accum, selector)
  master.foreach_shard(table, mapper_kernel, (src.id(), dst.id(), fn, args))
  return dst


def map_inplace(table, fn, *args):
  src = table
  dst = src
  table.master.foreach_shard(table, mapper_kernel, (src.id(), dst.id(), fn, args))
  return dst
  
def key_mapper(k, v):
  yield k, 1
  
def keys(src):
  return map_items(src, key_mapper)