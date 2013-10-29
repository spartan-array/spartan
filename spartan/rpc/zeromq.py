'''ZeroMQ socket implementation.'''
import atexit

import collections

import os
import threading
import time
import sys
import traceback

import zmq

from .common import Group, SocketBase


class Socket(SocketBase):
  # __slots__ = ['_zmq', '_hostport', '_out', '_in', '_addr', '_closed', '_shutdown', '_lock']
  def __init__(self, ctx, sock_type, hostport):
    # util.log('New socket...')
    self._zmq = ctx.socket(sock_type)
    self.addr = hostport
    self._out = collections.deque()
    self._closed = True
    self._shutdown = False
    self._lock = threading.RLock()

  def in_poll_loop(self):
    return threading.current_thread() == POLLER

  def __repr__(self):
    return 'Socket(%s)' % (self.addr)

  def flush(self):
    self.handle_write()

  def close(self, *args):
    if self.in_poll_loop():
      POLLER.remove(self)
      self.handle_close()
    else:
      self._shutdown = True
      POLLER.close(self)

  def send(self, msg):
    assert not self._closed
    self._out.append(msg)
    POLLER.modify(self, zmq.POLLIN | zmq.POLLOUT)

  def zmq(self):
    return self._zmq

  def recv(self):
    assert not self._closed
    frames = self._zmq.recv_multipart(copy=False, track=False)
    assert len(frames) == 1
    return frames[0]

  def connect(self):
    assert self._closed
    self._closed = False
    #    util.log('Connecting: %s:%d' % self.addr)
    self._zmq.connect('tcp://%s:%s' % self.addr)
    POLLER.add(self, zmq.POLLIN)

  @property
  def port(self):
    return self.addr[1]

  @property
  def host(self):
    return self.addr[0]

  def handle_close(self):
    self._closed = True
    self._zmq.close()
    del self._zmq

  def handle_write(self):
    with self._lock:
      while self._out:
        next = self._out.popleft()
        if isinstance(next, Group):
          # util.log('Sending %s', next.args)
          self._zmq.send_multipart(next, copy=False)
        else:
          # util.log('Sending %s', next)
          self._zmq.send(next, copy=False)

      POLLER.modify(self, zmq.POLLIN)

  def handle_read(self, socket):
    self._handler(socket)


class ServerSocket(Socket):
  def __init__(self, ctx, sock_type, hostport):
    Socket.__init__(self, ctx, sock_type, hostport)
    self.addr = hostport

  def send(self, msg):
    '''Send ``msg`` to a remote client.

    :param msg: `.Group`, with the first element being the destination to send to.
    '''
    Socket.send(self, msg)

  def bind(self):
    assert self._closed
    #    util.log('Binding...')
    self._closed = False
    host, port = self.addr
    if port == -1:
      self.addr = (host, self._zmq.bind_to_random_port('tcp://%s' % host))
    else:
      self._zmq.bind('tcp://%s:%d' % self.addr)
    POLLER.add(self, zmq.POLLIN)

  def handle_read(self, socket):
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
    self._poller.register(self._pipe[0], zmq.POLLIN)
    self._sockets = {}

    self._closing = {}
    self.setDaemon(True)

  def _run(self):
    self._running = True
    _poll = self._poller.poll

    while self._running:
      socks = dict(_poll(100))
      with self._lock:
        for fd, event in socks.iteritems():
          if fd == self._pipe[0]:
            os.read(fd, 1)
            continue

          if not fd in self._sockets:
            # util.log('Handler for %s disappeared...', fd)
            continue

          socket = self._sockets[fd]
          if event & zmq.POLLIN:
            socket.handle_read(socket)
          if event & zmq.POLLOUT:
            socket.handle_write()

        for socket in self._closing.keys():
          socket.handle_close()
        self._closing.clear()

  def close(self, socket):
    'Execute socket.handle_close() from within the polling thread.'
    with self._lock:
      self._closing[socket] = 1
      self.remove(socket)
      self.wakeup()

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
      self._poller.modify(socket.zmq(), direction)
      self.wakeup()

  def add(self, socket, direction):
    self._sockets[socket.zmq()] = socket
    with self._lock:
      self._poller.register(socket.zmq(), direction)
      self.wakeup()

  def remove(self, socket):
    with self._lock:
      self._poller.unregister(socket.zmq())
      del self._sockets[socket.zmq()]


def shutdown():
  POLLER.stop()

CTX = zmq.Context()
POLLER = ZMQPoller()
POLLER.start()

import atexit
atexit.register(shutdown)

def server_socket(addr):
  host, port = addr
  return ServerSocket(CTX, zmq.ROUTER, (host, port))


def client_socket(addr):
  host, port = addr
  return Socket(CTX, zmq.DEALER, (host, port))
