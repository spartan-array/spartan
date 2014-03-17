from libc.string cimport memcpy

cimport numpy as N
import numpy as N

from cpython cimport buffer
from cpython cimport string

ctypedef unsigned char byte

cdef extern from "Python.h":
  void PyBuffer_Release(Py_buffer*)

cdef Py_buffer get_buffer(object o):
  cdef Py_buffer b
  if buffer.PyObject_GetBuffer(o, &b, buffer.PyBUF_SIMPLE) != 0:
    raise Exception, 'Failed to get buffer.'
  b.readonly = 0
  return b

cdef class Buffer:
  cdef byte* base
  cdef int len
  cdef Py_buffer buffer
  #cdef object buf_array

  def __init__(self, v):
    if isinstance(v, int):
      v = N.ndarray(v, dtype=N.uint8)

    #self.buf_array = v
    self.buffer = get_buffer(v)
    self.base = <byte*>self.buffer.buf
    self.len = self.buffer.len

  def __dealloc__(self):
    PyBuffer_Release(&self.buffer)

  #cdef byte[:] view(self):
  #  return self.buf_array

  def __len__(self):
    return self.len

  cdef byte* data(self):
    return self.base

  cdef int size(self):
    return self.len

cdef class Writer:
  cdef int pos
  cdef Buffer buffer

  def __init__(self, int default_size=1000):
    self.reserve(default_size)
    self.pos = 0

  cdef reserve(self, int num):
    if self.buffer is None or self.buffer.size() - self.pos < num:
      new_len = 2 * (num + self.pos)
      new_buffer = Buffer(new_len)
      if self.buffer is not None:
        memcpy(new_buffer.data(), self.buffer.data(), self.pos)
      self.buffer = new_buffer

  #cpdef object getvalue(self):
  #  return self.buffer.view()[:self.pos]
  
  cpdef object getvalue(self):
    return self.buffer.data()[0:self.pos]
  
  cpdef int write(self, v):
    cdef Py_buffer data = get_buffer(v)
    self.write_bytes(< byte *> data.buf, data.len)
    PyBuffer_Release(&data)
    return self.pos

  cdef void write_bytes(self, byte * ptr, int len):
    self.reserve(len)
    memcpy(self.buffer.data() + self.pos, ptr, len)
    self.pos += len

  cdef void write_str(self, str v):
    self.write_int(len(v))
    self.write_bytes(v, len(v))

  cdef void write_int(self, int v):
    self.write_bytes(< byte *> & v, 4)

  cdef void write_array(self, N.ndarray v):
    self.write_bytes(v.dtype, 4)
    self.write_bytes(<byte*> N.PyArray_DATA(v), N.PyArray_NBYTES(v))

  cpdef int tell(self):
    return self.pos

  cpdef seek(self, int pos):
    self.pos = pos


cdef class Reader:
  cdef Buffer buffer
  cdef int pos
  def __init__(self, object v):
    self.buffer = Buffer(v)
    self.pos = 0

  cpdef str read(self, len):
    cdef count = min(self.buffer.size() - self.pos, len)
    cdef str v = string.PyString_FromStringAndSize(<char*>self.buffer.base + self.pos, count)
    self.pos += count
    return v

  cpdef int readinto(self, out):
    cdef Py_buffer b = get_buffer(out)
    cdef int count = min(self.buffer.size() - self.pos, b.len)
    memcpy(b.buf, self.buffer.data() + self.pos, count)
    self.pos += count
    return self.pos

  cpdef str readline(self):
    cdef byte* v = self.buffer.data() + self.pos
    cdef int end = self.buffer.size() - self.pos
    cdef i = 0
    while i < end:
      if v[i] == '\n':
        break
      i += 1

    if i < end:
      self.pos += i + 1
      return string.PyString_FromStringAndSize(<char*>v, i + 1)
    else:
      self.pos = end
      return string.PyString_FromString('')

  cpdef int tell(self):
    return self.pos

  cpdef seek(self, int pos):
    self.pos = pos