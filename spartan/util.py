#!/usr/bin/env python
import atexit

import cStringIO
import collections
from contextlib import contextmanager
import logging
from math import ceil
import os
from os.path import basename
import socket
import sys
import threading
import time
import traceback
import numpy as np

from spartan.config import FLAGS, BoolFlag


FLAGS.add(BoolFlag('dump_timers', default=False))

HOSTNAME = socket.gethostname()
PID = os.getpid()

LOGGING_CONFIGURED = False


def _setup_logger():
  global LOGGING_CONFIGURED
  if logging.root is None:
    raise Exception, 'Log attempt before logging was configured.'

  logging.RootLogger.findCaller = findCaller
  LOGGING_CONFIGURED = True


def log_debug(*args, **kw):
  if not LOGGING_CONFIGURED: _setup_logger()
  kw['extra'] = {'hostname': HOSTNAME, 'pid': PID}
  logging.debug(*args, **kw)


def log_info(*args, **kw):
  if not LOGGING_CONFIGURED: _setup_logger()
  kw['extra'] = {'hostname': HOSTNAME, 'pid': PID}
  logging.info(*args, **kw)


def log_warn(*args, **kw):
  if not LOGGING_CONFIGURED: _setup_logger()
  kw['extra'] = {'hostname': HOSTNAME, 'pid': PID}
  logging.warn(*args, **kw)


def log_error(*args, **kw):
  if not LOGGING_CONFIGURED: _setup_logger()
  kw['extra'] = {'hostname': HOSTNAME, 'pid': PID}
  logging.error(*args, **kw)


def log_fatal(*args, **kw):
  if not LOGGING_CONFIGURED: _setup_logger()
  kw['extra'] = {'hostname': HOSTNAME, 'pid': PID}
  logging.fatal(*args, **kw)


def findCaller(obj):
  f = sys._getframe(5)
  co = f.f_code
  filename = os.path.normcase(co.co_filename)
  return co.co_filename, f.f_lineno, co.co_name


class TimeHelper(object):
  def __init__(self, timer, name):
    self.timer = timer
    self.name = name

  def __enter__(self):
    self.start = time.time()

  def __exit__(self, exc_type, exc_val, exc_tb):
    end = time.time()
    self.timer.add(self.name, end - self.start)


class Timer(object):
  def __init__(self):
    self._times = collections.defaultdict(float)
    self._counts = collections.defaultdict(int)

  def add(self, name, t):
    self._counts[name] += 1
    self._times[name] += t

  def dump(self):
    for name, elapsed in self._times.iteritems():
      print os.getpid(), name, self._counts[name], elapsed

  def __getattr__(self, key):
    return TimeHelper(self, key)

TIMER = Timer()
def _dump_timer():
  if FLAGS.dump_timers:
    TIMER.dump()


atexit.register(_dump_timer)


def flatten(lst, depth=1, unique=False):
  if depth == 0:
    return lst

  out = []
  for item in lst:
    if isinstance(item, (list, set)):
      out.extend(flatten(item, depth=depth - 1))
    else:
      out.append(item)

  if unique:
    out = list(set(out))

  return out


def timeit(f, name=None):
  '''
  Run ``f`` and return (time_taken, result).

  :param f:
  :param name:
  '''
  st = time.time()
  res = f()
  ed = time.time()

  return ed - st, res


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


class Timer(object):
  def __init__(self):
    self.elapsed = 0

  def start(self):
    self.st = time.time()

  def stop(self):
    self.elapsed += time.time() - self.st

  def __enter__(self):
    #self.start()
    pass

  def __exit__(self, exc_type, exc_val, exc_tb):
    #self.stop()
    pass


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
  def all_eq(a, b, tolerance=0):
    if isinstance(a, np.ndarray) and isinstance(b, np.ndarray):
      assert a.shape == b.shape, 'Mismatched shapes: %s %s' % (a.shape, b.shape)
      if tolerance == 0:
        assert np.all(a == b), 'Failed: \n%s\n ==\n%s' % (a, b)
      else:
        assert np.all(np.abs(a - b) < tolerance), 'Failed: \n%s\n ==\n%s' % (a, b)
      return

    if np.isscalar(a) or np.isscalar(b):
      if tolerance == 0:
        assert a == b, 'Failed: \n%s\n ==\n%s' % (a, b)
      else:
        import math
        assert abs(a - b) < tolerance, 'Failed: \n%s\n ==\n%s' % (a, b)
      return

    assert is_iterable(a), (a, b)
    assert is_iterable(b), (a, b)

    for i, j in zip(a, b):
      assert i == j, 'Failed: \n%s\n ==\n%s' % (a, b)

  @staticmethod
  def all_close(a, b):
    assert isinstance(a, np.ndarray) and isinstance(b, np.ndarray)
    assert a.shape == b.shape, 'Mismatched shapes: %s %s' % (a.shape, b.shape)
    assert np.allclose(a, b), 'Failed: \n%s close to \n%s' % (a, b)
    return

  @staticmethod
  def float_close(a, b):
    '''Test floating point equality.'''
    Assert.all_close(np.array(a), np.array(b))

  @staticmethod
  def eq(a, b, fmt='', *args):
    assert a == b, 'Failed: %s == %s (%s)' % (a, b, fmt % args)

  @staticmethod
  def ne(a, b, fmt='', *args):
    assert a != b, 'Failed: %s != %s (%s)' % (a, b, fmt % args)

  @staticmethod
  def gt(a, b, fmt='', *args):
    assert a > b, 'Failed: %s > %s (%s)' % (a, b, fmt % args)

  @staticmethod
  def lt(a, b, fmt='', *args):
    assert a < b, 'Failed: %s < %s (%s)' % (a, b, fmt % args)

  @staticmethod
  def ge(a, b, fmt='', *args):
    assert a >= b, 'Failed: %s >= %s (%s)' % (a, b, fmt % args)

  @staticmethod
  def le(a, b, fmt='', *args):
    assert a <= b, 'Failed: %s <= %s (%s)' % (a, b, fmt % args)

  @staticmethod
  def true(expr):
    assert expr, 'Failed: %s == True' % (expr)

  @staticmethod
  def iterable(expr):
    assert is_iterable(expr), 'Not iterable: %s' % expr

  @staticmethod
  def isinstance(expr, klass):
    assert isinstance(expr, klass), 'Failed: isinstance(%s, %s) [type = %s]' % (expr, klass, type(expr))

  @staticmethod
  def no_duplicates(collection):
    d = collections.defaultdict(int)
    for item in collection:
      d[item] += 1

    bad = [(k, v) for k, v in d.iteritems() if v > 1]
    assert len(bad) == 0, 'Duplicates found: %s' % bad

  @staticmethod
  def not_null(expr):
    assert expr is not None, expr

  @staticmethod
  def raises_exception(exception, function, *args, **kwargs):
    try:
      function(*args, **kwargs)
    except exception:
      return
    assert False, '%s expected, no error was raised.' % exception.__name__


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


def calc_tile_hint(array, axis=0):
  if isinstance(array, tuple):
    tile_hint = list(array)
  else:
    tile_hint = list(array.shape)
  tile_hint[axis] = divup(tile_hint[axis], FLAGS.num_workers)
  return tile_hint


def is_iterable(x):
  return hasattr(x, '__iter__')


def is_lambda(fn):
  """Return True if ``fn`` is a lambda expression.

  For some reason testing against LambdaType does not work correctly.
  """
  return fn.__name__ == '<lambda>'


def as_list(x):
  if isinstance(x, list): return x
  if is_iterable(x): return list(x)

  return [x]


def get_core_mapping():
  '''
  Read /proc/cpuinfo and return a dictionary mapping from:

  ``processor_id -> (package, core)``

  '''
  lines = open('/proc/cpuinfo').read().strip().split('\n')
  package_id = core_id = None
  id = 0
  cpus = {}
  for l in lines:
    if l.startswith('physical id'):
      package_id = int(l.split(':')[1].strip())

    if l.startswith('core_id'):
      core_id = int(l.split(':')[1].strip())

    if package_id is not None and core_id is not None:
      cpus[id] = (package_id, core_id)

  return cpus


def memoize(f):
  '''Decorator.

  Cache outputs of ``f``; repeated calls with the same arguments will be
  served from the cache.
  '''
  _cache = {}

  def wrapped(*args):
    if args not in _cache:
      _cache[args] = f(*args)
    return _cache[args]

  wrapped.__name__ = f.__name__
  wrapped.__doc__ = f.__doc__
  return wrapped


def copy_docstring(source_function):
  '''
  Decorator.

  Copy the docstring from ``source_function`` to this function.
  '''
  def _decorator(func):
    source_doc = source_function.__doc__
    if func.__doc__ is None:
      func.__doc__ = source_doc
    else:
      func.__doc__ = source_doc + '\n\n' + func.__doc__
    return func
  return _decorator


class FileHelper(object):
  def __init__(self, **files):
    # FileHelper(x=open('x'), y=open('y'...)
    self._files = files
    for k, v in files.iteritems():
      setattr(self, k, v)

  def __enter__(self):
    return self

  def __exit__(self, exc_type, exc_val, exc_tb):
    for v in self._files.values():
     v.close()
