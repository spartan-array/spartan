'''
Simple RPC library.

The :class:`.Client` and :class:`.Server` classes here work with
sockets which should implement the :class:`.Socket` interface.
'''
import weakref
import cPickle

import sys
import threading
import time
import traceback

from . import msg, core
from .. import util

RPC_ID = xrange(1000000000).__iter__()

CLIENT_PENDING = weakref.WeakKeyDictionary()
SERVER_PENDING = weakref.WeakKeyDictionary()

DEFAULT_TIMEOUT = 100
def set_default_timeout(seconds):
  global DEFAULT_TIMEOUT
  DEFAULT_TIMEOUT = seconds
  util.log_info('Set default timeout to %s seconds.', DEFAULT_TIMEOUT)


class SocketBase(object):
  def send(self, blob): pass
  def recv(self): pass
  def flush(self): pass
  def close(self): pass

  def register_handler(self, handler):
    'A handler() is called in response to read requests.'
    self._handler = handler

  # client
  def connect(self): pass

  # server
  def bind(self): pass


def capture_exception(exc_info=None):
  if exc_info is None:
    exc_info = sys.exc_info()
  tb = traceback.format_exception(*exc_info)
  return msg.Exception(py_exc=''.join(tb).replace('\n', '\n:: '))


class Group(tuple):
  pass


class PendingRequest(object):
  '''An outstanding RPC request.

  Call done(result) when a method is finished processing.
  '''
  def __init__(self, socket, rpc_id):
    self.socket = socket
    self.rpc_id = rpc_id
    self.created = time.time()
    self.finished = False

    SERVER_PENDING[self] = 1

  def done(self, result=None):
    # util.log_info('RPC finished in %.3f seconds' % (time.time() - self.created))
    if result is None:
      result = msg.EMPTY

    self.finished = True
    assert isinstance(result, msg.Message), 'Bad result type %s' % result
    header = { 'rpc_id' : self.rpc_id, 'klass' : result.__class__ }
    # util.log_info('Finished %s, %s', self.socket.addr, self.rpc_id)
    w = core.Writer()
    cPickle.dump(header, w, -1)
    result.encode(w)
    self.socket.send(w.getvalue())

  def __del__(self):
    if not self.finished:
      logging.error('PendingRequest.done() not called before destruction (likely due to an exception.)')
      self.done(result=msg.Exception('done() not called on request.'))


class RemoteException(Exception):
  '''Wrap a uncaught remote exception.'''
  def __init__(self, tb):
    self._tb = tb

  def __repr__(self):
    return 'RemoteException:\n' + self._tb

  def __str__(self):
    return repr(self)

class FnFuture(object):
  '''Chain ``fn`` to the given future.

  ``self.wait()`` return ``fn(future.wait())``.
  '''
  def __init__(self, future, fn):
    self.future = future
    self.fn = fn
    self.result = None

  def wait(self):
    result = self.future.wait()
    # util.log_info('Applying %s to %s', self.fn, result)
    self.result = self.fn(result)
    return self.result

class Future(object):
  def __init__(self, addr, rpc_id):
    self.addr = addr
    self.rpc_id = rpc_id
    self.have_result = False
    self.result = None
    self.finished_fn = None
    self._cv = threading.Condition()
    self._start = time.time()
    self._deadline = time.time() + DEFAULT_TIMEOUT

    CLIENT_PENDING[self] = 1

  def _set_result(self, result):
    self._cv.acquire()
    self.have_result = True

    if self.finished_fn is not None:
      self.result = self.finished_fn(result)
    else:
      self.result = result

    self._cv.notify()
    self._cv.release()

  def timed_out(self):
    return self._deadline < time.time()

  def wait(self):
    self._cv.acquire()
    while not self.have_result and not self.timed_out():
      # use a timeout so that ctrl-c works.
      self._cv.wait(timeout=0.1)
    self._cv.release()

#    util.log_info('Result from %s in %f seconds.', self.addr, time.time() - self._start)

    if not self.have_result and self.timed_out():
      util.log_info('timed out!')
      raise Exception('Timed out on remote call (%s %s)', self.addr, self.rpc_id)

    if isinstance(self.result, msg.Exception):
      raise RemoteException(self.result.py_exc)
    return self.result

  def on_finished(self, fn):
    return FnFuture(self, fn)


class DummyFuture(object):
  def __init__(self, base=None):
    self.v = base

  def wait(self):
    return self.v

DUMMY_FUTURE = DummyFuture()

class FutureGroup(list):
  def wait(self):
    return [f.wait() for f in self]

def wait_for_all(futures):
  return [f.wait() for f in futures]


class Server(object):
  def __init__(self, socket):
    self._socket = socket
    self._socket.register_handler(self.handle_read)
    self._methods = {}
    self._running = False
    self.register_method('diediedie', self._diediedie, msg.Empty, msg.Empty)

  def _diediedie(self, handle, req):
    handle.done(None)
    self._socket.flush()
    self.shutdown()

  @property
  def addr(self):
    return self._socket.addr

  def serve(self):
    self.serve_nonblock()
    while self._running:
      time.sleep(0.1)

  def serve_nonblock(self):
#    util.log_info('Running.')
    self._running = True
    self._socket.bind()

  def register_method(self, name, fn, req_class, resp_class):
    self._methods[name] = (fn, req_class, resp_class)

  def handle_read(self, socket):
    data = socket.recv()
    reader = core.Reader(data)
    header = cPickle.load(reader)

    # util.log_info('Starting: %s %s', self._socket.addr, header['rpc_id'])
    handle = PendingRequest(socket, header['rpc_id'])
    name = header['method']
    try:
      fn, req_class, resp_class = self._methods[name]
    except KeyError:
      handle.done(capture_exception())
      return

    try:
      req = msg.Message.decode(reader)
      fn(handle, req)
    except:
      util.log_info('Caught exception in handler.', exc_info=1)
      handle.done(capture_exception())

  def shutdown(self):
    self._running = 0
    self._socket.close()
    del self._socket


class ProxyMethod(object):
  def __init__(self, client, method):
    self.client = client
    self.socket = client._socket
    self.method = method

  def __call__(self, request=None):
    rpc_id = RPC_ID.next()

    if request is None:
      request = msg.EMPTY

    header = { 'method' : self.method, 'rpc_id' : rpc_id }

    f = Future(self.socket.addr, rpc_id)
    self.client._futures[rpc_id] = f
    w = core.Writer()
    cPickle.dump(header, w, 0)
    if isinstance(request, msg.Message):
      request.encode(w)
#      util.collect_time(lambda: request.encode(w), 'Encoding time')
    else:
      w.write(request)
#    util.log_info('Sending %d bytes for %s', len(serialized), self.method)
#    if len(serialized) > 800000:
#      util.log_info('%s::\n %s; \n\n\n %s', self.method, ''.join(traceback.format_stack()), request)

    self.socket.send(w.getvalue())
    return f

class Client(object):
  def __init__(self, socket):
    self._socket = socket
    self._socket.register_handler(self.handle_read)
    self._socket.connect()
    self._futures = {}

  def __reduce__(self, *args, **kwargs):
    raise cPickle.PickleError('Not pickleable.')

  def __enter__(self):
    return self

  def __exit__(self, exc_type, exc_val, exc_tb):
    self.close()

  def close(self):
    self._socket.close()

  def __getattr__(self, method_name):
    return ProxyMethod(self, method_name)

  def handle_read(self, socket):
    data = socket.recv()
    import numpy as N
    reader = core.Reader(N.frombuffer(data, dtype=N.uint8))
    header = cPickle.load(reader)
    resp = header['klass'].decode(reader)
    rpc_id = header['rpc_id']
    f = self._futures[rpc_id]
    f._set_result(resp)
    del self._futures[rpc_id]

  def close(self):
    self._socket.close()
