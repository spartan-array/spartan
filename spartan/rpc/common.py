'''
Simple RPC library.

The :class:`.Client` and :class:`.Server` classes here work with
sockets which should implement the :class:`.Socket` interface.
'''
from cPickle import PickleError
import collections
import weakref
import sys
import threading
import time
import traceback
import types
import cStringIO
import cPickle

from .. import cloudpickle, util, core
from ..node import Node

CLIENT_PENDING = weakref.WeakKeyDictionary()
SERVER_PENDING = weakref.WeakKeyDictionary()

NO_RESULT = object()

DEFAULT_TIMEOUT = 100

def set_default_timeout(seconds):
  global DEFAULT_TIMEOUT
  DEFAULT_TIMEOUT = seconds
  util.log_info('Set default timeout to %s seconds.', DEFAULT_TIMEOUT)


class RPCException(object):
  __metaclass__ = Node
  _members = ['py_exc']

class PickledData(object):
  '''
  Helper class: indicates that this message has already been pickled,
  and should be sent as is, rather than being re-pickled.
  '''
  __metaclass__ = Node
  _members = ['data']

class SocketBase(object):
  def send(self, blob): assert False
  def recv(self): assert False
  def flush(self): assert False
  def close(self): assert False

  def register_handler(self, handler):
    'A handler() is called in response to read requests.'
    self._handler = handler

  # client
  def connect(self): assert False

  # server
  def bind(self): assert False


def capture_exception(exc_info=None):
  if exc_info is None:
    exc_info = sys.exc_info()
  tb = traceback.format_exception(*exc_info)
  return RPCException(py_exc=''.join(tb).replace('\n', '\n:: '))


class Group(tuple):
  pass

def serialize_to(obj, writer):
  #serialization.write(obj, writer)
  try:
    pickled = cPickle.dumps(obj, -1)
    writer.write(pickled)
  except (PickleError, TypeError):
    #util.log_warn('CPICKLE failed: %s (%s)', sys.exc_info(), obj)
    writer.write(cloudpickle.dumps(obj, -1))

def serialize(obj):
  #x = cStringIO.StringIO()
  #serialization.write(obj, x)
  #return x.getvalue()
  try:
    return cPickle.dumps(obj, -1)
  except (PickleError, TypeError):
    return cloudpickle.dumps(obj, -1)

def read(f):
  return cPickle.load(f)

class PendingRequest(object):
  '''An outstanding RPC request on the server.

  Call done(result) when finished to send result back to client.
  '''
  def __init__(self, socket, rpc_id):
    self.socket = socket
    self.rpc_id = rpc_id
    self.created = time.time()
    self.finished = False
    self.result = NO_RESULT

    SERVER_PENDING[self] = 1

  def wait(self):
    while self.result is NO_RESULT:
      time.sleep(0.1)
    return self.result

  def exception(self):
    self.done(capture_exception())

  def done(self, result=None):
    # util.log_info('RPC finished in %.3f seconds' % (time.time() - self.created))
    self.finished = True
    self.result = result

    assert self.socket is not None
    header = { 'rpc_id' : self.rpc_id }
    #util.log_info('Finished %s, %s', self.socket.addr, self.rpc_id)
    w = cStringIO.StringIO()
    cPickle.dump(header, w, -1)
    serialize_to(result, w)
    self.socket.send(w.getvalue())

  def __del__(self):
    if not self.finished:
      util.log_error('PendingRequest.done() not called before destruction (likely due to an exception.)')
      self.done(result=RPCException(py_exc='done() not called on request.'))


class RemoteException(Exception):
  '''Wrap a uncaught remote exception.'''
  def __init__(self, tb):
    self._tb = tb

  def __repr__(self):
    return 'RemoteException:\n' + self._tb

  def __str__(self):
    return repr(self)


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

  def done(self, result=None):
    #util.log_info('Result... %s %s', self.addr, self.rpc_id)
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
      self._cv.wait(timeout=1)
      if time.time() - self._start > 2:
        util.log_info('Waiting... %s %s', self.addr, self.rpc_id)
    self._cv.release()

#    util.log_info('Result from %s in %f seconds.', self.addr, time.time() - self._start)

    if not self.have_result and self.timed_out():
      util.log_info('timed out!')
      raise Exception('Timed out on remote call (%s %s)', self.addr(), self.rpc_id)

    if isinstance(self.result, RPCException):
      raise RemoteException(self.result.py_exc)
    return self.result

  def on_finished(self, fn):
    return FnFuture(self, fn)


class FnFuture(object):
  '''Chain ``fn`` to the given future.

  ``self.wait()`` return ``fn(future.wait())``.
  '''
  def __init__(self, future, fn):
    self.future = future
    self.fn = fn
    self.result = None

  def wait(self):
    return self.fn(self.future.wait())


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
    self._timers = collections.defaultdict(util.Timer)
    self._running = False
    self.register_method('diediedie', self._diediedie)


  def timings(self):
    return '\n'.join(['%s: %f' % (m, self._timers[m].get())
                      for m in self._methods.keys()])

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
      time.sleep(1)

  def serve_nonblock(self):
#    util.log_info('Running.')
    self._running = True
    self._socket.bind()

  def register_object(self, obj):
    for name in dir(obj):
      if name.startswith('__'): continue
      fn = getattr(obj, name)
      if isinstance(fn, types.MethodType):
        self.register_method(name, fn)
  
  def register_method(self, name, fn):
    self._methods[name] = fn

  def handle_read(self, socket):
    #util.log_info('Reading...')

    data = socket.recv()
    reader = cStringIO.StringIO(data)
    header = cPickle.load(reader)

    #util.log_info('Call[%s] %s %s', header['method'], self._socket.addr, header['rpc_id'])
    handle = PendingRequest(socket, header['rpc_id'])
    name = header['method']

    try:
      fn = self._methods[name]
    except KeyError:
      handle.exception()
      return

    try:
      req = read(reader)
      result = fn(req, handle)
      assert result is None, 'non-None result from RPC handler (use handle.done())'
    except:
      util.log_info('Caught exception in handler.', exc_info=1)
      handle.exception()

  def shutdown(self):
    self._running = 0
    self._socket.close()
    del self._socket


class ProxyMethod(object):
  def __init__(self, client, method):
    self.client = client
    self.method = method

  def __call__(self, request=None):
#    if len(serialized) > 800000:
#      util.log_info('%s::\n %s; \n\n\n %s', self.method, ''.join(traceback.format_stack()), request)
    return self.client.send(self.method, request)

class Client(object):
  def __init__(self, socket):
    self._socket = socket
    self._socket.register_handler(self.handle_read)
    self._socket.connect()
    self._futures = {}
    self._lock = threading.Lock()
    self._rpc_id = xrange(10000000).__iter__()

  def __reduce__(self, *args, **kwargs):
    raise cPickle.PickleError('Not pickleable.')

  def send(self, method, request):
    with self._lock:
      rpc_id = self._rpc_id.next()
      header = { 'method' : method, 'rpc_id' : rpc_id }

      w = cStringIO.StringIO()
      cPickle.dump(header, w, -1)
      if isinstance(request, PickledData):
        w.write(request.data)
      else:
        serialize_to(request, w)

      data = w.getvalue()
      f = Future(self.addr(), rpc_id)
      self._futures[rpc_id] = f
      #util.log_info('Send %s, %s', self.addr(), rpc_id)
      self._socket.send(data)
      return f

  def addr(self):
    return self._socket.addr

  def close(self):
    self._socket.close()

  def __getattr__(self, method_name):
    return ProxyMethod(self, method_name)

  def handle_read(self, socket):
    data = socket.recv()
    reader = cStringIO.StringIO(data)
    header = cPickle.load(reader)
    resp = read(reader)
    #resp = cPickle.load(reader)
    rpc_id = header['rpc_id']
    f = self._futures[rpc_id]
    f.done(resp)
    del self._futures[rpc_id]

  def close(self):
    self._socket.close()


def forall(clients, method, request):
  '''Invoke ``method`` with ``request`` for each client in ``clients``

  ``request`` is only serialized once, so this is more efficient when
  targeting multiple workers with the same data.

  Returns a future wrapping all of the requests.
  '''
  futures = []
  pickled = PickledData(data=serialize(request))
  for c in clients:
    futures.append(getattr(c, method)(pickled))

  return FutureGroup(futures)

