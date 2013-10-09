#!/usr/bin/env python

from contextlib import contextmanager
from math import ceil
from os.path import basename
import cStringIO
import collections
import os
import select
import sys
import threading
import time
import traceback

from wrap import log, log_info, log_debug, log_warn, log_error

class FileWatchdog(threading.Thread):
  """Watchdog for a file (typically `sys.stdin` or `sys.stdout`).

  When the file closes, terminate the process.
  (This occurs when an ssh connection is terminated, for example.)
  """
  def __init__(self, file_handle=sys.stdin, on_closed=lambda: os._exit(1)):
    threading.Thread.__init__(self, name='WatchdogThread')
    self.setDaemon(True)
    self.file_handle = file_handle
    self.log = sys.stderr
    self.on_closed = on_closed

  def run(self):
    f = [self.file_handle]
    while 1:
      r, w, x = select.select(f, f, f, 1.0)
      if r:
        try:
          print >>self.log, 'Watchdog: file closed.  Shutting down.'
        except:
          pass
        
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
  st = time.time()
  res = f()
  ed = time.time()
  if name is None:
    name = f
  log('Operation %s completed in %.3f seconds', name, ed - st)
  return res

def collect_time(f, name=None, timings={}):
  st = time.time()
  res = f()
  ed = time.time()

  if not '_last_log' in timings:
    timings['_last_log'] = ed

  if not name in timings:
    timings[name] = 0
  timings[name] += ed - st

  if ed - timings['_last_log'] > 5:
    for k, v in timings.items():
      log('Timing: %s %s', k, v)
    timings['_last_log'] = time.time()

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
  print('%3.5f:: %s' % (ed - st, name))

class Timer(object):
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
  @staticmethod
  def all_eq(a, b):
    import numpy
    if hasattr(a, 'shape') and hasattr(b, 'shape'):
      assert a.shape == b.shape, 'Mismatched shapes: %s %s' % (a.shape, b.shape)
      
    assert numpy.all(a == b), 'Failed: \n%s\n ==\n%s' % (a, b)
  
  @staticmethod
  def eq(a, b): assert (a == b), 'Failed: %s == %s' % (a, b)
  
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
    log('TRACE: >> %s with args: %s %s', fn, args, kw)
    result = fn(*args, **kw)
    log('TRACE: << %s (%s)', fn, result)
    return result
  return tracer
   
   
def rtype_check(typeclass):
  '''Function decorator to check return type.
  
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
  
  
def count_calls(fn):
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
  return int(ceil(float(a) / b))
