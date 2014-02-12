'''
Simple RPC library.

The :class:`.Client` and :class:`.Server` classes here work with
sockets which should implement the :class:`.Socket` interface.
'''
from cPickle import PickleError
import cPickle
import cStringIO
import collections
import os
import pickle
import sys
import threading
import time
import traceback
import types
import weakref
from .. import cloudpickle, util, core
from ..node import Node, node_type
from . import serialization


CLIENT_PENDING = weakref.WeakKeyDictionary()
SERVER_PENDING = weakref.WeakKeyDictionary()

NO_RESULT = object()

#DEFAULT_TIMEOUT = 1200
DEFAULT_TIMEOUT = 60
WARN_THRESHOLD = 10

def set_default_timeout(seconds):
  global DEFAULT_TIMEOUT
  DEFAULT_TIMEOUT = seconds
  util.log_info('Set default timeout to %s seconds.', DEFAULT_TIMEOUT)


@node_type
class RPCException(object):
  _members = ['py_exc']

@node_type
class PickledData(object):
  '''
  Helper class: indicates that this message has already been pickled,
  and should be sent as is, rather than being re-pickled.
  '''
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


def capture_exception(exc_info=None):
  if exc_info is None:
    exc_info = sys.exc_info()
  tb = traceback.format_exception(*exc_info)
  return RPCException(py_exc=''.join(tb).replace('\n', '\n:: '))


class Group(tuple):
  pass

def serialize_to(obj, writer):
  serialization.write(obj, writer)
  #writer.write(serialize(obj))

def serialize(obj):
  #w = cStringIO.StringIO()
  w = serialization.Writer()
  serialization.write(obj, w)
  return w.getvalue()
  
  #util.log_info('Pickling: %s', obj)  
  #try:
  #  return cPickle.dumps(obj, -1)
  #except (pickle.PicklingError, PickleError, TypeError):
  #  #print >>sys.stderr, 'Failed to cPickle: %s' % obj
  #  return cloudpickle.dumps(obj, -1)
    
def read(f):
  return serialization.read(f)
  #return cPickle.load(f)
# 
#   st = time.time()
#   start_pos = f.tell()
#   result = cPickle.load(f)
#   ed = time.time()
#   end_pos = f.tell()
#   
#   if ed - st > 0.1:
#     util.log_warn('Slow to load! %s' % result)
#     with open('./slow-pickle.%d' % os.getpid(), 'w') as pickle_out:
#       f.seek(start_pos)
#       pickle_out.write(f.read(end_pos - start_pos))
#     
#   return result

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
      time.sleep(0.01)
    return self.result
  
  def exception(self):
    self.done(capture_exception())

  def done(self, result=None):
    self.finished = True
    self.result = result

    assert self.socket is not None
    header = { 'rpc_id' : self.rpc_id }
    
    #util.log_info('Finished %s, %s', self.socket.addr, self.rpc_id)
    #w = cStringIO.StringIO()
    w = serialization.Writer()
    cPickle.dump(header, w, -1)
    serialize_to(result, w)
    self.socket.send(w.getvalue())

  def __del__(self):
    if not self.finished:
      util.log_error('PendingRequest.done() not called before destruction (likely due to an exception.)')
      self.done(result=RPCException(py_exc='done() not called on request.'))


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


class Future(object):
  def __init__(self, addr, rpc_id, timeout=None):
    self.addr = addr
    self.rpc_id = rpc_id
    self.have_result = False
    self.result = None
    self.finished_fn = None
    self._cv = threading.Condition()
    self._start = time.time()
    self._finish = time.time() + 1000000
    if timeout is None:
      timeout = DEFAULT_TIMEOUT
    self._deadline = time.time() + timeout
    CLIENT_PENDING[self] = 1

  def done(self, result=None):
    #util.log_info('Result... %s %s', self.addr, self.rpc_id)
    self._cv.acquire()
    self.have_result = True
    self._finish = time.time()

    if self.finished_fn is not None:
      self.result = self.finished_fn(result)
    else:
      self.result = result

    self._cv.notify()
    self._cv.release()

  def timed_out(self):
    return self._deadline < time.time()

  def elapsed_time(self):
    return min(time.time(), self._finish) - self._start
  
  def __repr__(self):
    return 'Future(%s:%d) [%s]' % (self.addr, self.rpc_id, self.elapsed_time())
  
  def wait(self):
    #if self.addr is not None:
    #  util.log_info('Waiting for %s', self)
    
    self._cv.acquire()
    while not self.have_result and not self.timed_out():
      # use a timeout so that ctrl-c works.
      self._cv.wait(timeout=1)
      
      if self.elapsed_time() > WARN_THRESHOLD:
        util.log_info('Waiting for result from %s RPC: %s', self.addr, self.rpc_id)
    self._cv.release()

#    util.log_info('Result from %s in %f seconds.', self.addr, time.time() - self._start)

    if not self.have_result and self.timed_out():
      util.log_info('timed out!')
      raise TimeoutException('Timed out on remote call (%s %s)' % (self.addr, self.rpc_id))

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
    results = []
    for f in self:
      #util.log_info('Waiting for %s', f)
      results.append(f.wait())
    return results

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
      time.sleep(0.1)

  def serve_nonblock(self):
#    util.log_info('Running.')
    self._running = True
    self._socket.listen()

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
    #reader = cStringIO.StringIO(data)
    reader = serialization.Reader(data)
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
    util.log_debug('Server going down...')
    self._running = 0
    self._socket.close()
    del self._socket


class ProxyMethod(object):
  def __init__(self, client, method):
    self.client = client
    self.method = method

  def __call__(self, request=None, timeout=None):
#    if len(serialized) > 800000:
#      util.log_info('%s::\n %s; \n\n\n %s', self.method, ''.join(traceback.format_stack()), request)
    return self.client.send(self.method, request, timeout)

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

  def send(self, method, request, timeout):
    with self._lock:
      rpc_id = self._rpc_id.next()
      header = { 'method' : method, 'rpc_id' : rpc_id }

      #w = cStringIO.StringIO()
      w = serialization.Writer()
      cPickle.dump(header, w, -1)
      if isinstance(request, PickledData):
        w.write(request.data)
      else:
        serialize_to(request, w)

      data = w.getvalue()
      f = Future(self.addr(), rpc_id, timeout)
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
    #reader = cStringIO.StringIO(data)
    reader = serialization.Reader(data)
    header = cPickle.load(reader)
    resp = read(reader)
    #resp = cPickle.load(reader)
    rpc_id = header['rpc_id']
    f = self._futures[rpc_id]
    f.done(resp)
    del self._futures[rpc_id]

def forall(clients, method, request, timeout=None):
  '''Invoke ``method`` with ``request`` for each client in ``clients``

  ``request`` is only serialized once, so this is more efficient when
  targeting multiple workers with the same data.

  Returns a future wrapping all of the requests.
  '''
  futures = []
  pickled = PickledData(data=serialize(request))
  for c in clients:
    futures.append(getattr(c, method)(pickled, timeout=timeout))

  return FutureGroup(futures)

