'''ZeroMQ socket implementation.'''
import cProfile
import collections
import os
import threading
import socket
import traceback
import sys
import fcntl
import time
import zmq
from .common import Group
from spartan import util
from zmq.eventloop import zmqstream, ioloop 
from sys import stderr
import Queue
from rlock import FastRLock

CLOSED = 0
CONNECTING = 1
CONNECTED = 2

#for client socket, we have one polling loop per thread.
_loop = threading.local()

def get_threadlocal_loop():
  global _loop
  if not hasattr(_loop, "val"):
    _loop.val = ioloop.IOLoop()
  return _loop.val

class ZMQLoop(object):
  '''  
  A subset functions of tornado.ioloop, used only for server socket
  to send data and receive data in polling thread.    
  '''
  
  '''These constants are used by the zmqstream class'''  
  # Constants from the epoll module
  _EPOLLIN = 0x001
  _EPOLLPRI = 0x002
  _EPOLLOUT = 0x004
  _EPOLLERR = 0x008
  _EPOLLHUP = 0x010
  _EPOLLRDHUP = 0x2000
  _EPOLLONESHOT = (1 << 30)
  _EPOLLET = (1 << 31)
  # Our events map exactly to the epoll events
  NONE = 0
  READ = _EPOLLIN
  WRITE = _EPOLLOUT
  ERROR = _EPOLLERR | _EPOLLHUP

  def __init__(self):
    self._poller = ioloop.ZMQPoller()
    self._handler = {}
    self._running = False
    self._running_thread = None
    self._callback = None
    self._callback_lock = FastRLock() 

  def add_handler(self, fd, handler, events):
    ''' This function will be called by ZMQStream to add handler'''
    self._poller.register(fd, zmq.POLLIN | self.ERROR)    
    self._handler[fd] = handler 

  def update_handler(self, fd, events):
    ''' This function will be called by ZMQStream to update handler'''
    self._poller.modify(fd, events | self.ERROR)  

  def remove_handler(self, fd):
    ''' This function will be called by ZMQStream to remove handler'''
    self._handlers.pop(fd, None)  
  
  def start(self):
    MAX_TIMEOUT = 0.01
    self._running = True
    #record the running thread
    self._running_thread = threading.current_thread()
    _poll_time = 0.001
    _poll = self._poller.poll
    
    while self._running:
      socks = dict(_poll(_poll_time))
      if len(socks) == 0:
        _poll_time = min(_poll_time * 2, MAX_TIMEOUT)
      else:
        _poll_time = 0.001
      
      for fd, events in socks.iteritems():
        handler = self._handler[fd]
        handler(None, events)
     
      with self._callback_lock:
        callback = self._callback
        self._callback = None 
      
      #call callback function
      if callback: callback()

  def stop(self):
    self._running = False  
     
  def running_thread(self):
    ''' query the running thread '''
    return self._running_thread 

  def add_callback(self, callback):
    ''' add callback funtion, it will be called in next loop '''
    with self._callback_lock:
      self._callback = callback 


class Socket(object):
  def __init__(self, ctx, sock_type, hostport, loop=None):
    self._zmq = ctx.socket(sock_type)
    self.addr = hostport
    self._event_loop = loop or get_threadlocal_loop() 
    self._stream = zmqstream.ZMQStream(self._zmq, self._event_loop)
    #Register this socket with the ZMQ stream to handle incoming messages.
    self._stream.on_recv(self.on_recv)
    self._status = CLOSED
    
  def __repr__(self):
    return 'Socket(%s)' % ((self.addr,))
  
  def closed(self):
    return self._status == CLOSED

  def close(self, *args):
    self._zmq.close()
    self._status = CLOSED

  def send(self, msg):
    if isinstance(msg, Group):
      self._zmq.send_multipart(msg, copy=False)
    else:
      self._zmq.send(msg, copy=False)

  def on_recv(self, msg):
    self._handler(msg[0])

  def connect(self):
    self._zmq.connect('tcp://%s:%s' % self.addr)
    self._status = CONNECTED

  @property
  def port(self):
    return self.addr[1]

  @property
  def host(self):
    return self.addr[0]
  
  def register_handler(self, handler):
    self._handler = handler

class ServerSocket(Socket):
  def __init__(self, ctx, sock_type, hostport, loop = ZMQLoop()):
    Socket.__init__(self, ctx, sock_type, hostport, loop)
    self._listen = False
    self.addr = hostport
    self._out = collections.deque() 
    self._out_lock = FastRLock()
    self.bind()

  def listen(self):
    self._listen = True
  
  def handle_write(self):
    ''' 
    Thiss function is called from inside of the poll thread.
    It sends out any messages which were enqueued by other threads. 
    '''
    with self._out_lock:
      while self._out:
        msg = self._out.popleft()
        if isinstance(msg, Group):
          self._zmq.send_multipart(msg, copy=False)
        else:
          self._zmq.send(msg, copy=False)

  def send(self, msg):
    if threading.current_thread() != self._event_loop.running_thread(): 
      #if the this function is not called in polling thread, put the message in queue
      #to guarantee thread safety, otherwise send the message directly.
      with self._out_lock:
        self._out.append(msg)
        self._event_loop.add_callback(self.handle_write)
    else:
      if isinstance(msg, Group):
        self._zmq.send_multipart(msg, copy=False)
      else:
        self._zmq.send(msg, copy=False)

  def bind(self):
    host, port = self.addr
    host = socket.gethostbyname(host)
    util.log_debug('Binding... %s', (host, port))
    if port == -1:
      self.addr = (host, self._zmq.bind_to_random_port('tcp://%s' % host))
    else:
      try:
        self._zmq.bind('tcp://%s:%d' % (host, port))
      except zmq.ZMQError:
        util.log_info('Failed to bind (%s, %d)' % (host, port))
        raise
    
  def on_recv(self, msg):
    if self.listen == False:
      return
    source, rest = msg[0], msg[1:]
    stub_socket = StubSocket(source, self, rest)
    self._handler(stub_socket)


class StubSocket(object):
  '''Handles a single read from a client'''
  def __init__(self, source, socket, data):
    self._out = collections.deque()
    self.source = source
    self.socket = socket
    assert isinstance(data, list)
    assert len(data) == 1
    self.data = data[0]

  @property
  def addr(self):
    return self.socket.addr

  def recv(self):
    assert self.data is not None, 'Tried to read twice from stub socket.'
    res = self.data
    self.data = None
    return res

  def send(self, req):
    if isinstance(req, Group):
      req = Group([self.source] + list(req))
    else:
      req = Group((self.source, req))
    self.socket.send(req)


def server_socket(addr):
  host, port = addr
  return ServerSocket(zmq.Context.instance(), zmq.ROUTER, (host, port))

def server_socket_random_port(host):
  return ServerSocket(zmq.Context.instance(), 
                      zmq.ROUTER, 
                      (host, -1))

def client_socket(addr):
  host, port = addr
  return Socket(zmq.Context.instance(), zmq.DEALER, (host, port))
