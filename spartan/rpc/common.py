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
from ..node import Node
from . import serialization
from traits.api import PythonValue
from . import serialization, serialization_buffer
from spartan.util import TIMER

NO_RESULT = object()
DEFAULT_TIMEOUT = 60
WARN_THRESHOLD = 10
_rpc_id_generator = xrange(10000000).__iter__()

def set_default_timeout(seconds):
  global DEFAULT_TIMEOUT
  DEFAULT_TIMEOUT = seconds
  util.log_info('Set default timeout to %s seconds.', DEFAULT_TIMEOUT)

class RPCException(Node):
    py_exc = PythonValue

class PickledData(Node):
    data = PythonValue

def capture_exception(exc_info=None):
  if exc_info is None:
    exc_info = sys.exc_info()
  tb = traceback.format_exception(*exc_info)
  return RPCException(py_exc=''.join(tb).replace('\n', '\n:: '))

class Group(tuple):
  pass

def serialize_to(obj, writer):
  pos = writer.tell()
  try:
    cPickle.dump(obj, writer, -1)
  except (pickle.PicklingError, PickleError, TypeError):
    writer.seek(pos)
    cloudpickle.dump(obj, writer, -1)
    
def serialize(obj):
  try:
    return cPickle.dumps(obj, -1)
  except (pickle.PicklingError, PickleError, TypeError):
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
    
    w = serialization_buffer.Writer()
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
  def __init__(self, addr, rpc_id, timeout=None, poller=None):
    self.addr = addr
    self.rpc_id = rpc_id
    self.have_result = False
    self.result = None
    self._poller = poller 
    if timeout is None:
      timeout = DEFAULT_TIMEOUT
    self._timeout = timeout 

  def done(self, result=None):
    self.have_result = True
    self.result = result

  def __repr__(self):
    return 'Future(%s:%d)' % (self.addr, self.rpc_id)
  
  def wait(self):
    while not self.have_result:
      socks = dict(self._poller.poll(self._timeout * 1000))
      for fd, events in socks.iteritems():
        # Here we only care about read. We send message directly.
        self._poller._sockets[fd].handle_read()
  
      if len(socks) == 0:
        # It means timeout.
        util.log_info('timed out!')
        raise TimeoutException('Timed out on remote call (%s %s)' % (self.addr, self.rpc_id))

    if isinstance(self.result, RPCException):
      raise RemoteException(self.result.py_exc)
    return self.result


class BroadcastFuture(object):
  def __init__(self, rpc_id, n_jobs, timeout=None, poller=None):
    self.rpc_id = rpc_id
    self.have_all_results = False
    self.results = [] 
    self._poller = poller 
    self._start = time.time()
    self._finish = time.time() + 1000000
    if timeout is None:
      timeout = DEFAULT_TIMEOUT
    self._n_jobs = n_jobs
    self._timeout = timeout

  def done(self, result=None):
    self._n_jobs -= 1
    self.results.append(result)
    if self._n_jobs == 0:
      self.have_all_results = True

  def __repr__(self):
    return 'Future(%d) [%s]' % (self.rpc_id, self.elapsed_time())
  
  def wait(self):
    while not self.have_all_results:
      socks = dict(self._poller.poll(self._timeout * 1000))
      for fd, events in socks.iteritems():
        # Here we only care about read. We send message directly.
        self._poller._sockets[fd].handle_read()
      
      if len(socks) == 0:
        # It means timeout.
        util.log_info('timed out!')
        raise TimeoutException('Timed out on remote call (%s %s)' % (self.addr, self.rpc_id))
    
    for result in self.results:
      if isinstance(result, RPCException):
        raise RemoteException(result.py_exc)
    return self.results


class FutureGroup(list):
  def wait(self):
    results = []
    for f in self:
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
    ''' wait util polling thread stops'''
    self.serve_nonblock().join()

  def serve_nonblock(self):
    # We start a new thread to run the polling loop for the server socket.
    poll_thread = threading.Thread(target=self._socket._event_loop.start)
    poll_thread.daemon=True
    poll_thread.start()
    self._socket.listen()
    return poll_thread

  def register_object(self, obj):
    for name in dir(obj):
      if name.startswith('__'): continue
      fn = getattr(obj, name)
      if isinstance(fn, types.MethodType):
        self.register_method(name, fn)
  
  def register_method(self, name, fn):
    self._methods[name] = fn

  def handle_read(self, socket):
    data = socket.recv()
    reader = serialization_buffer.Reader(data)
    header = cPickle.load(reader)
    
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
    self._socket._event_loop.stop()

class ProxyMethod(object):
  def __init__(self, client, method):
    self.client = client
    self.method = method

  def __call__(self, request=None, timeout=None):
    return self.client.send(self.method, request, timeout)

class Client(object):
  def __init__(self, socket):
    self._socket = socket
    self._socket.register_handler(self.handle_read)
    self._socket.connect()
    self._futures = {}

  def __reduce__(self, *args, **kwargs):
    raise cPickle.PickleError('Not pickleable.')

  def send(self, method, request, timeout):
    rpc_id = _rpc_id_generator.next()
    header = { 'method' : method, 'rpc_id' : rpc_id }

    w = serialization_buffer.Writer()
    cPickle.dump(header, w, -1)
    if isinstance(request, PickledData):
      w.write(request.data)
    else:
      serialize_to(request, w)

    data = w.getvalue()
    f = Future(self.addr(), rpc_id, timeout, self._socket._poller)
    self._futures[rpc_id] = f
    self._socket.send(data)
    return f
  
  def send_raw(self, data, future):
    '''
    This function is used to send serialized data.
    Now only used in 'forall' function, so we can 
    serialize the messages once and send the serialized 
    message to all clients.
    '''
    self._futures[future.rpc_id] = future
    self._socket.send(data)     
 
  def addr(self):
    return self._socket.addr

  def __getattr__(self, method_name):
    return ProxyMethod(self, method_name)

  def handle_read(self, data):
    reader = serialization_buffer.Reader(data)
    header = cPickle.load(reader)
    resp = read(reader)
    rpc_id = header['rpc_id']
    f = self._futures[rpc_id]
    f.done(resp)
    del self._futures[rpc_id]

def forall(clients, method, request, timeout=None):
  ''' 
  Invoke ``method`` with ``request`` for each client in ``clients``

  ``request`` and header is only serialized once, and we only create one
  future object, so this is more efficient when targeting multiple workers 
  with the same data.

  Returns a BroadcastFuture wrapping all of the requests. 
 
  '''  
  n_jobs = len(clients)  
  rpc_id = _rpc_id_generator.next()
  fgroup = BroadcastFuture(rpc_id, 
                            n_jobs, 
                            timeout=timeout, 
                            poller=clients[0]._socket._poller) 

  with TIMER.serial_once:
    # Only serialize the header and body once for all the clients.
    header = { 'method' : method, 'rpc_id' : rpc_id }
    w = serialization_buffer.Writer()
    cPickle.dump(header, w, -1)
    serialize_to(request, w)
    data = w.getvalue()
  
  with TIMER.master_loop:
    for c in clients:
      c.send_raw(data=data, future=fgroup)
  return fgroup 


from zeromq import client_socket

# Use thread-locals to create one "real" client per thread
_clients = threading.local()

# ZeroMQ sockets are _not_ thread-safe.  
# To avoid locking overhead/deadlock issues, requests are proxied through
# a unique, thread-local client for each thread.  This has the potential to create
# many clients (if a thread were spawned per request), but works well in practice.
# We use thread-locals instead of adding locks is that this design is simpler than 
# adding lock. For adding lock, we not only need to consider sending conflict, but also
# need to consider receiving/receiving or sending/receiving conflict.
class ThreadLocalClient(object):
  def __init__(self, host, port):
    self.host = host
    self.port = port

  def __getattr__(self, method):
    if not hasattr(_clients, "val"):
      _clients.val = {}
    
    if self not in _clients.val:
      socket = client_socket((self.host, self.port))
      _clients.val[self] = Client(socket)
  
    return getattr(_clients.val[self], method)
