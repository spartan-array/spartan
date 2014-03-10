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
from .common import Group, SocketBase
from . import rlock
from spartan import util

POLLER = None
POLLER_LOCK = rlock.FastRLock()
PROFILER = None

def poller():
  global POLLER
  with POLLER_LOCK:
    if POLLER is None:
      util.log_debug('Started poller.. %s %s', os.getpid(), __file__)
      POLLER = ZMQPoller()
      POLLER.start()
    return POLLER

def in_poll_loop():
  return threading.current_thread() == poller()

socket_count = collections.defaultdict(int)

CLOSED = 0
CONNECTING = 1
CONNECTED = 2

class Socket(SocketBase):
  __slots__ = ['_zmq', '_hostport', '_out', '_in', 'addr', '_status', '_lock']

  def __init__(self, ctx, sock_type, hostport):
    socket_count[os.getpid()] += 1
    #util.log_info('New socket... %s.%s -> %s [%s]',
    #              socket.gethostname(), os.getpid(),  hostport, socket_count[os.getpid()])
    #s = ''.join(traceback.format_stack())
    #print >>sys.stderr, s
    self._zmq = ctx.socket(sock_type)
    self.addr = hostport
    self._out = collections.deque()
    self._lock = rlock.FastRLock()

    self._status = CLOSED
  def __repr__(self):
    return 'Socket(%s)' % ((self.addr,))

  def closed(self):
    return self._status == CLOSED

  def close(self, *args):
    util.log_debug('Closing socket: %s (%s)', self.addr, self.closed())
    if self.closed():
      return

    with self._lock:
      poller().queue_close(self)

  def send(self, msg):
    assert not self.closed()

    with self._lock:
      #util.log_info('SEND %s', len(msg))
      self._out.append(msg)
      poller().modify(self, zmq.POLLIN | zmq.POLLOUT)

  def zmq(self):
    return self._zmq

  def recv(self):
    assert not self.closed()
    with self._lock:
      frames = self._zmq.recv_multipart(copy=False, track=False)
      assert len(frames) == 1
      return frames[0]

  def connect(self):
    assert self.closed()
    with self._lock:
      #util.log_info('Connecting: %s:%d' % self.addr)
      self._zmq.connect('tcp://%s:%s' % self.addr)
      poller().add(self, zmq.POLLIN)
      self._status = CONNECTED

  @property
  def port(self):
    return self.addr[1]

  @property
  def host(self):
    return self.addr[0]

  def handle_close(self):
    assert not self.closed()

    self._status = CLOSED
    self._zmq.close()
    self._zmq = None
    util.log_debug('Socket closed: %s', self.addr)

  def handle_write(self):
    with self._lock:
      if not self._out:
        return

      while self._out:
        next = self._out.popleft()
        if isinstance(next, Group):
          #util.log_info('Sending group. %s', len(next))
          self._zmq.send_multipart(next, copy=False)
        else:
          #util.log_info('Sending %s', len(next))
          self._zmq.send(next, copy=False)

      poller().modify(self, zmq.POLLIN)

  def handle_read(self, socket):
    self._handler(socket)


class ServerSocket(Socket):
  def __init__(self, ctx, sock_type, hostport):
    Socket.__init__(self, ctx, sock_type, hostport)
    self._listen = False
    self.addr = hostport
    self.bind()
  
  def listen(self):
    self._listen = True

  def send(self, msg):
    '''Send ``msg`` to a remote client.

    :param msg: `.Group`, with the first element being the destination to send to.
    '''
    Socket.send(self, msg)

  def bind(self):
    assert self.closed()
    self._status = CONNECTED
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

    poller().add(self, zmq.POLLIN)

  def handle_read(self, socket):
    if self._listen == False:
      #ignore the message if it's not been started.
      return

    packet = self._zmq.recv_multipart(copy=False, track=False)
    source, rest = packet[0], packet[1:]
    stub_socket = StubSocket(source, self, rest)
    self._handler(stub_socket)

  def zmq(self):
    return self._zmq


class StubSocket(SocketBase):
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
    # print 'Result!', res
    return res

  def send(self, req):
    if isinstance(req, Group):
      req = Group([self.source] + list(req))
    else:
      req = Group((self.source, req))
    self.socket.send(req)


class ZMQPoller(threading.Thread):
  def __init__(self):
    threading.Thread.__init__(self, name='zmq.PollingThread', target=self._run)

    self._poller = zmq.Poller()
    self._lock = threading.RLock()

    self._pipe = os.pipe()
    fcntl.fcntl(self._pipe[0], fcntl.F_SETFL, os.O_NONBLOCK)
    fcntl.fcntl(self._pipe[1], fcntl.F_SETFL, os.O_NONBLOCK)

    self._poller.register(self._pipe[0], zmq.POLLIN)
    self._sockets = {}
    self.profiler = None

    self._running = False

    self._to_add = []
    self._to_close = []
    self._to_mod = []
    self.daemon = True

  def enable_profiling(self):
    self.profiler = cProfile.Profile()

  def disable_profiling(self):
    self.profiler = None

  def _run(self):
    self._running = True
    _poll = self._poller.poll
    _poll_time = 1
    MAX_TIMEOUT = 10

    while self._running:
      #if self.profiler: self.profiler.disable()
      socks = dict(_poll(_poll_time))
      if self.profiler:
        self.profiler.enable()

      if len(socks) == 0:
        _poll_time = min(_poll_time * 2, MAX_TIMEOUT)
      else:
        _poll_time = 1

      #if self.profiler: 
      #  self.profiler.enable()

      #util.log_info('%s', self._sockets)
      for fd, event in socks.iteritems():
        if fd == self._pipe[0]:
          os.read(fd, 10000)
          continue

        if not fd in self._sockets:
          continue

        s = self._sockets[fd]
        if event & zmq.POLLIN: s.handle_read(s)
        if event & zmq.POLLOUT: s.handle_write()

      with self._lock:
        for s, dir in self._to_add:
          self._sockets[s.zmq()] = s
          self._poller.register(s.zmq(), dir)

        for s, dir in self._to_mod:
          self._poller.register(s.zmq(), dir)

        for s in self._to_close:
          #util.log_info('Removing... %s (%s)', id(s), s.addr)
          self._poller.unregister(s.zmq())
          del self._sockets[s.zmq()]
          s.handle_close()

        del self._to_mod[:]
        del self._to_add[:]
        del self._to_close[:]

    util.log_info("ZMQ poller stopped.")

  def stop(self):
    if not self._running:
      return

    self._running = False
    self.wakeup()
    if threading.current_thread() != self:
      self.join()

  def wakeup(self):
    os.write(self._pipe[1], 'x')

  def modify(self, socket, direction):
    with self._lock:
      assert socket.zmq() in self._sockets, socket.zmq()
      #util.log_info('Modifying... %s', socket.zmq())
      self._to_mod.append((socket, direction))
      self.wakeup()

  def add(self, socket, direction):
    if in_poll_loop():
      self._sockets[socket.zmq()] = socket
      self._poller.register(socket.zmq(), direction)
      return

    with self._lock:
      assert not socket.zmq() in self._sockets, socket.zmq()
      #util.log_info('Adding... %s', socket.zmq())
      self._to_add.append((socket, direction))
      self.wakeup()

    # wait until we get added
    while not socket.zmq() in self._sockets:
      time.sleep(0.0001)

  def queue_close(self, socket):
    'Execute socket.handle_close() from within the polling thread.'
    with self._lock:
      assert socket.zmq() in self._sockets
      self._to_close.append(socket)
      self.wakeup()

def shutdown():
  poller().stop()

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
