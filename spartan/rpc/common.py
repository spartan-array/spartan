'''
Simple Fast RPC library.

The :class:`.Client` and :class:`.Server` classes here work with
sockets which should implement the :class:`.Socket` interface.
'''
from cPickle import PickleError
import cPickle
import pickle
import sys
import traceback
from . import future, cloudpickle
from .. import util

def set_default_timeout(seconds):
  future.set_default_timeout(seconds)
  util.log_info('Set default timeout to %s seconds.', future.DEFAULT_TIMEOUT)

def capture_exception(exc_info=None):
  if exc_info is None:
    exc_info = sys.exc_info()
  tb = traceback.format_exception(*exc_info)
  return ''.join(tb).replace('\n', '\n:: ')

class TimeoutException(Exception):
  '''Wrap a timeout exception.'''
  def __init__(self, tb):
    self._tb = tb

  def __repr__(self):
    return 'TimeoutException:' + self._tb

  def __str__(self):
    return repr(self)

class RemoteException(Exception):
  '''Wrap a uncaught remote exception.'''
  def __init__(self, tb):
    self._tb = tb

  def __repr__(self):
    return 'RemoteException:\n' + self._tb

  def __str__(self):
    return repr(self)

def serialize_to(obj, writer):
  pos = writer.tell()
  try:
    cPickle.dump(obj, writer, -1)
  except (ImportError, pickle.PicklingError, PickleError, TypeError):
    writer.seek(pos)
    cloudpickle.dump(obj, writer, -1)

def serialize(obj):
  try:
    return cPickle.dumps(obj, -1)
  except (pickle.PicklingError, PickleError, TypeError):
    return cloudpickle.dumps(obj, -1)

def read(f):
  return cPickle.loads(f)


