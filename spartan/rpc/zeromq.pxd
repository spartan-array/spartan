'''ZeroMQ socket implementation.'''
from rlock cimport FastRLock

cdef class Socket:
  cdef public object _zmq
  cdef public object addr
  cdef public object _out
  cdef public object _status
  cdef public object _handler
  cdef public FastRLock _lock
  cpdef handle_read(self, Socket socket)

cdef class ServerSocket(Socket):
  cdef int _listen
  cpdef handle_read(self, Socket socket)

cdef class StubSocket:
  '''Handles a single read from a client'''
  cdef object _out
  cdef object source
  cdef object data
  cdef Socket socket
