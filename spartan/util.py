#!/usr/bin/env python

from contextlib import contextmanager
import logging
from math import ceil
from os.path import basename
import collections
import os
import select
import sys
import threading
import time
import traceback

import cStringIO


def log_info(msg, *args):
  msg = msg % args
  print >>sys.stderr, msg


log_debug = logging.debug
log_info = logging.info
log_warn = logging.warn
log_error = logging.error
log_fatal = logging.fatal

def findCaller(obj):
  f = sys._getframe(4)
  co = f.f_code
  filename = os.path.normcase(co.co_filename)
  return co.co_filename, f.f_lineno, co.co_name

root = logging.getLogger()
logging.RootLogger.findCaller = findCaller


class FileWatchdog(threading.Thread):
  """Watchdog for a file (typically `sys.stdin` or `sys.stdout`).

  When the file closes, terminate the process.
  (This occurs when an ssh connection is terminated, for example.)
  """
  def __init__(self, file_handle=sys.stdin, on_closed=lambda: os._exit(1)):
    '''
    
    :param file_handle:
    :param on_closed:
    '''
    threading.Thread.__init__(self, name='WatchdogThread')
    self.setDaemon(True)
    self.file_handle = file_handle
    self.on_closed = on_closed

  def run(self):
    f = [self.file_handle]
    while 1:
      r, w, x = select.select(f, f, f, 1.0)
      if r:
        self.on_closed()
        return
      
      time.sleep(0.1)

def flatten(lst, depth=1):
  if depth == 0:
    return lst

  out = []
  for item in lst:
    if isinstance(item, (list, set)): out.extend(flatten(item, depth=depth - 1))
    else: out.append(item)
  return out


def timeit(f, name=None):
  '''
  Run ``f`` and log the amount of time taken.
   
  :param f:
  :param name:
  '''
  st = time.time()
  res = f()
  ed = time.time()
  if name is None:
    name = f
  log_info('Operation %s completed in %.3f seconds', name, ed - st)
  return res

@contextmanager
def timer_ctx(name='Operation'):
  '''Context based timer:

  Usage::

    with timer_ctx('LoopOp'):
      for i in range(10):
        my_op()

  '''
  st = time.time()
  yield
  ed = time.time()
  log_info('%3.5f:: %s' % (ed - st, name))

class EZTimer(object):
  '''Lazy timer.

  Prints elapsed time when destroyed.
  '''
  def __init__(self, name):
    self.name = name
    self.st = time.time()

  def __del__(self):
    print('%3.5f:: %s' % (time.time() - self.st, self.name))

def dump_stacks(out):
  '''Dump the stacks of all threads.'''
  id_to_name = dict([(th.ident, th.name) for th in threading.enumerate()])
  thread_stacks = collections.defaultdict(list)

  for thread_id, stack in sys._current_frames().items():
    code = []
    for filename, lineno, name, line in traceback.extract_stack(stack):
      if line is None: line = ''
      code.append('%s:%d (%s): %s' % (basename(filename), lineno, name, line.strip()))
    thread_stacks['\n'.join(code)].append(thread_id)

  for stack, thread_ids in thread_stacks.iteritems():
    print >> out, 'Thread %d(%s)' % (thread_ids[0], id_to_name.get(thread_ids[0], ''))
    if len(thread_ids) > 1:
      print >> out, '... and %d more' % (len(thread_ids) - 1)
    print >> out, stack
    print >> out

def stack_signal():
  out = cStringIO.StringIO()
  dump_stacks(out)
  print >> sys.stderr, out.getvalue()
  with open('/tmp/%d.stacks' % os.getpid(), 'w') as f:
    print >> f, out.getvalue()


class Assert(object):
  '''Assertion helper functions.
  
  ::
  
    a = 'foo'
    b = 'bar'
    
    Assert.eq(a, b) 
    # equivalent to:
    # assert a == b, 'a == b failed (%s vs %s)' % (a, b) 
  '''
  @staticmethod
  def all_eq(a, b):
    import numpy
    if hasattr(a, 'shape') and hasattr(b, 'shape'):
      assert a.shape == b.shape, 'Mismatched shapes: %s %s' % (a.shape, b.shape)
      
    assert numpy.all(a == b), 'Failed: \n%s\n ==\n%s' % (a, b)
  
  @staticmethod
  def eq(a, b, msg=''): 
    assert (a == b), 'Failed: %s == %s (%s)' % (a, b, msg)
  
  @staticmethod
  def ne(a, b): assert (a == b), 'Failed: %s != %s' % (a, b)
  
  @staticmethod
  def gt(a, b): assert (a > b), 'Failed: %s > %s' % (a, b)
  
  @staticmethod
  def lt(a, b): assert (a < b), 'Failed: %s < %s' % (a, b)
  
  @staticmethod
  def ge(a, b): assert (a >= b), 'Failed: %s >= %s' % (a, b)
  
  @staticmethod
  def le(a, b): assert (a <= b), 'Failed: %s <= %s' % (a, b)
  
  @staticmethod
  def true(expr): assert expr, 'Failed: %s == True' % (expr)
 
  @staticmethod
  def iterable(expr): assert iterable(expr), 'Not iterable: %s' % expr
   
  @staticmethod
  def isinstance(expr, klass): 
    assert isinstance(expr, klass), 'Failed: isinstance(%s, %s) [type = %s]' % (expr, klass, type(expr))
  
  @staticmethod
  def no_duplicates(collection):
    d = collections.defaultdict(int)
    for item in collection:
      d[item] += 1
    
    bad = [(k,v) for k, v in d.iteritems() if v > 1]
    assert len(bad) == 0, 'Duplicates found: %s' % bad
  
 
def trace_fn(fn):
  '''Function decorator: log on entry and exit to ``fn``.'''
  def tracer(*args, **kw):
    log_info('TRACE: >> %s with args: %s %s', fn, args, kw)
    result = fn(*args, **kw)
    log_info('TRACE: << %s (%s)', fn, result)
    return result
  return tracer
   
   
def rtype_check(typeclass):
  '''Function decorator to check return type.
  
  Usage::
  
    @rtype_check(int)
    def fn(x, y, z):
      return x + y
  '''
  
  def wrap(fn):
    def checked_fn(*args, **kw):  
      result = fn(*args, **kw)
      Assert.isinstance(result, typeclass)
      return result
    checked_fn.__name__ = 'checked_' + fn.__name__
    checked_fn.__doc__ = fn.__doc__
    return checked_fn
  return wrap

def synchronized(fn):
  '''
  Decorator: execution of this function is serialized by an `threading.RLock`.
  :param fn:
  '''
  lock = threading.RLock()
  def _fn(*args, **kw):
    with lock:
      return fn(*args, **kw)

  if hasattr(fn, '__name__'):
    _fn.__name__ = fn.__name__
  else:
    _fn.__name__ = 'unnamed'

  return _fn
  
def count_calls(fn):
  '''
  Decorator: count calls to ``fn`` and print after each 100.
  :param fn:
  '''
  count = [0]
  def wrapped(*args, **kw):
    count[0] += 1
    if count[0] % 100 == 0: print count[0], fn.__name__
    return fn(*args, **kw)
  
  wrapped.__name__ = 'counted_' + fn.__name__
  wrapped.__doc__ = fn.__doc__
  return wrapped

def join_tuple(tuple_a, tuple_b):
  return tuple(list(tuple_a) + list(tuple_b))

def divup(a, b):
  if isinstance(a, tuple):
    return tuple([divup(ta, b) for ta in a])
  
  return int(ceil(float(a) / b))

def iterable(x):
  return hasattr(x, '__iter__')

def as_list(x):
  if isinstance(x, list): return x
  if iterable(x): return list(x)
  
  return [x]
