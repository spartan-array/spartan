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
from . import serialization, rlock
from traits.api import PythonValue
from . import serialization, serialization_buffer, rlock
from ..util import TIMER
cimport zeromq
from zeromq cimport Socket, ServerSocket, StubSocket
from rlock cimport FastRLock
import copy

#cdef extern from "time.h":
#  cdef struct timespec:
#    long int tv_sec
#    long int tv_nsec

#  int clock_gettime(int timerid, timespec *value)
 
cdef extern from "pthread.h":
  ctypedef struct pthread_mutex_t:
    pass
  
  ctypedef struct pthread_cond_t:
    pass
  
  int pthread_mutex_init(pthread_mutex_t *, void *)
  int pthread_mutex_destroy(pthread_mutex_t *)
  int pthread_mutex_lock(pthread_mutex_t *) nogil
  int pthread_mutex_unlock(pthread_mutex_t *) nogil
  
  int pthread_cond_init(pthread_cond_t *, void *)
  int pthread_cond_destroy(pthread_cond_t *)
  int pthread_cond_signal(pthread_cond_t *) nogil
  int pthread_cond_wait(pthread_cond_t *, pthread_mutex_t *) nogil
  #int pthread_cond_timedwait(pthread_cond_t *, pthread_mutex_t *, const timespec *) nogil

#CLOCK_REALTIME = 0
#CLOCK_MONOTONIC = 1
#DEF CLOCK_REALTIME = 0
#DEF CLOCK_MONOTONIC = 1    

CLIENT_PENDING = weakref.WeakKeyDictionary()
SERVER_PENDING = weakref.WeakKeyDictionary()

#These ids reserve for broad cast messages
_broadcast_rpc_id = xrange(1000000, 2000000).__iter__()

NO_RESULT = object()

#DEFAULT_TIMEOUT = 1200
cdef unsigned long DEFAULT_TIMEOUT = 60
cdef unsigned long  WARN_THRESHOLD = 10

cpdef set_default_timeout(unsigned long seconds):
  global DEFAULT_TIMEOUT
  DEFAULT_TIMEOUT = seconds
  util.log_info('Set default timeout to %s seconds.', DEFAULT_TIMEOUT)


class RPCException(Node):
  py_exc = PythonValue

class PickledData(Node):
  '''
  Helper class: indicates that this message has already been pickled,
  and should be sent as is, rather than being re-pickled.
  '''
  data = PythonValue


def capture_exception(exc_info=None):
  if exc_info is None:
    exc_info = sys.exc_info()
  tb = traceback.format_exception(*exc_info)
  return RPCException(py_exc=''.join(tb).replace('\n', '\n:: '))

def serialize_to(obj, writer):
  with util.TIMER.rpc_serialize:
    
    #writer.write(serialize(obj))
    pos = writer.tell()
    try:
      cPickle.dump(obj, writer, -1)
    except (pickle.PicklingError, PickleError, TypeError):
      writer.seek(pos)
      cloudpickle.dump(obj, writer, -1)
    """ 
    serialization.write(obj, writer)
    """
    
def serialize(obj):
  with util.TIMER.serialize:
    """
    w = serialization_buffer.Writer()
    serialization.write(obj, w)
    return w.getvalue()
    """
    try:
      return cPickle.dumps(obj, -1)
    except (pickle.PicklingError, PickleError, TypeError):
      return cloudpickle.dumps(obj, -1)
    
def read(f):
  with util.TIMER.rpc_read:
    #return serialization.read(f)
    return cPickle.load(f)

cdef class PendingRequest:
  '''An outstanding RPC request on the server.

  Call done(result) when finished to send result back to client.
  '''
  cdef StubSocket socket
  cdef long rpc_id
  cdef long created
  cdef object finished
  cdef object result

  def __init__(self, socket, rpc_id):
    self.socket = socket
    self.rpc_id = rpc_id
    self.created = time.time()
    self.finished = False
    self.result = NO_RESULT
    #SERVER_PENDING[self] = 1 
  
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

cdef class Condition:
  cdef pthread_mutex_t mutex
  cdef pthread_cond_t cond
  #cdef timespec now
 
  def __init__(self):
    pthread_mutex_init(&self.mutex, NULL)
    pthread_cond_init(&self.cond, NULL)
      
  def acquire(self):
    with nogil:
      pthread_mutex_lock(&self.mutex)
    
  def release(self):
    pthread_mutex_unlock(&self.mutex)
    
  def wait(self, timeout=1):
    #clock_gettime(CLOCK_REALTIME, &self.now)
    #self.now.tv_sec += timeout
    with nogil:
      #pthread_cond_timedwait(&self.cond, &self.mutex, &self.now)
      pthread_cond_wait(&self.cond, &self.mutex)
        
  def notify(self):
    pthread_cond_signal(&self.cond)
      
  def __del__(self):
    pthread_mutex_destroy(&self.mutex)
    pthread_cond_destroy(&self.cond)

cdef class Future:
  cdef object addr
  cdef long rpc_id
  cdef object have_result
  cdef object result
  cdef object finished_fn
  cdef object _cv
  cdef long _start
  cdef long _finish
  cdef long _deadline

  def __init__(self, addr, rpc_id, timeout=None):
    with util.TIMER.future_init:
      self.addr = addr
      self.rpc_id = rpc_id
      self.have_result = False
      self.result = None
      self.finished_fn = None
      #self._cv = threading.Condition(lock=rlock.FastRLock())
      self._cv = Condition()
      self._start = time.time()
      self._finish = time.time() + 1000000
      if timeout is None:
        timeout = DEFAULT_TIMEOUT
      self._deadline = time.time() + timeout
    #CLIENT_PENDING[self] = 1

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
    cdef list results = []
    for f in self:
      #util.log_info('Waiting for %s', f)
      results.append(f.wait())
    return results

cdef class BcastFutureGroup:
  cdef public long rpc_id
  cdef object have_all_results
  cdef list results
  cdef object _cv
  cdef long _start
  cdef long _finish
  cdef long _deadline
  cdef int  _n_jobs

  def __init__(self, rpc_id, n_jobs, timeout=None):
    self.rpc_id = rpc_id
    self._n_jobs = n_jobs
    self.have_all_results = False
    self.results = []
    self._cv = Condition()
    self._start = time.time()
    self._finish = time.time() + 1000000
    if timeout is None:
      timeout = DEFAULT_TIMEOUT
    self._deadline = time.time() + timeout

  def done(self, result=None):
    self._cv.acquire()
    self._n_jobs -= 1
    self.results.append(result)
    #notify if all jobs are done 
    if self._n_jobs == 0:
      self.have_all_results = True
      self._finish = time.time()
      self._cv.notify()
    self._cv.release()

  def timed_out(self):
    return self._deadline < time.time()

  def elapsed_time(self):
    return min(time.time(), self._finish) - self._start
  
  def __repr__(self):
    return 'Future(id:%d) [%s]' % (self.rpc_id, self.elapsed_time())
  
  def wait(self):
    self._cv.acquire()
    while not self.have_all_results and not self.timed_out():
      # use a timeout so that ctrl-c works.
      self._cv.wait(timeout=1)
      
      if self.elapsed_time() > WARN_THRESHOLD:
        util.log_info('Waiting for RPC: %s', self.rpc_id)
    
    self._cv.release()
    if not self.have_all_results and self.timed_out():
      util.log_info('timed out!')
      raise TimeoutException('Timed out on remote call (%s)' % (self.rpc_id,))
    
    for result in self.results:
      if isinstance(result, RPCException):
        raise RemoteException(result.py_exc)
    return self.results


def wait_for_all(futures):
  return [f.wait() for f in futures]


cdef class Server:
  cdef Socket _socket
  cdef dict _methods
  cdef object _timers
  cdef object _running

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
      time.sleep(0.01)

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

  cpdef handle_read(self, StubSocket socket):
    #util.log_info('Reading...')
    data = socket.recv()
    #reader = cStringIO.StringIO(data)
    reader = serialization_buffer.Reader(data)
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
    #del self._socket


cdef class ProxyMethod:
  cdef Client client
  cdef object method

  def __init__(self, client, method):
    self.client = client
    self.method = method

  #raw_data is true means the request has already been serialized along with the header
  def __call__(self, request=None, timeout=None, raw_data=False, future=None):
    return self.client.send(self.method, request, timeout, raw_data, future)

cdef class Client:
  cdef Socket _socket
  cdef dict _futures
  cdef FastRLock _lock
  cdef object _rpc_id
  
  def __init__(self, socket):
    self._socket = socket
    self._socket.register_handler(self.handle_read)
    self._socket.connect()
    self._futures = {}
    self._lock = rlock.FastRLock()
    self._rpc_id = xrange(10000000).__iter__()

  def __reduce__(self, *args, **kwargs):
    raise cPickle.PickleError('Not pickleable.')

  def send(self, method, request, timeout, raw_data, future):
    with self._lock:
      if not raw_data:
        rpc_id = self._rpc_id.next()
        header = { 'method' : method, 'rpc_id' : rpc_id }
        w = serialization_buffer.Writer()
        
        if isinstance(request, PickledData):
          with util.TIMER.send_serialize:
            cPickle.dump(header, w, -1)
            w.write(request.data)
        else:
          cPickle.dump(header, w, -1)
          serialize_to(request, w)

        data = w.getvalue()
        f = Future(self.addr(), rpc_id, timeout)
        self._futures[rpc_id] = f
      else:
        f = future
        data = request
        self._futures[f.rpc_id] = f

      self._socket.send(data)
      return f

  def addr(self):
    return self._socket.addr

  def close(self):
    self._socket.close()

  def __getattr__(self, method_name):
    return ProxyMethod(self, method_name)

  cpdef handle_read(self, Socket socket):
    data = socket.recv()
    #reader = cStringIO.StringIO(data)
    reader = serialization_buffer.Reader(data)
    header = cPickle.load(reader)
    resp = read(reader)
    #resp = cPickle.load(reader)
    rpc_id = header['rpc_id']
    f = self._futures[rpc_id]
    f.done(resp)
    del self._futures[rpc_id]

def forall(clients, method, request, timeout=None):
  '''Invoke ``method`` with ``request`` for each client in ``clients``

  ``request`` and header is only serialized once, and we only create one
  future object, so this is more efficient when targeting multiple workers 
  with the same data.

  Returns a BcastFutureGroup wrapping all of the requests.
  '''
  if not isinstance(clients, list):
    clients = [c for c in clients]
  
  n_jobs = len(clients)  
  rpc_id = _broadcast_rpc_id.next()
  fgroup = BcastFutureGroup(rpc_id, n_jobs) 

  with TIMER.serial_once:
    #only serialize the header and body once.
    header = { 'method' : method, 'rpc_id' : rpc_id }
    w = serialization_buffer.Writer()
    cPickle.dump(header, w, -1)
    serialize_to(request, w)
    data = w.getvalue()
  
  with TIMER.master_loop:
    for c in clients:
      getattr(c, method)(data, timeout=timeout, raw_data=True, future=fgroup)
  
  return fgroup
