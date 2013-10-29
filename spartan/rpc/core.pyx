from libc.string cimport memcpy

cimport numpy as N
import numpy as N

from cpython cimport buffer 
from cpython cimport string
 
from cython.view cimport memoryview  
 
ctypedef unsigned char byte

cdef extern from "Python.h":
  object PyMemoryView_FromBuffer(Py_buffer *)
  object PyMemoryView_FromObject(object)
  object PyBuffer_FromMemory(void*, Py_ssize_t)
  object PyBuffer_FromReadWriteMemory(void*, Py_ssize_t)
  void PyBuffer_Release(Py_buffer*)
  
cpdef class Shard(object):
  def __init__(self, id):
    self.id = id
    self.owner = -1
    self.data = {}

cpdef class Table(object):
  def __init__(self, id, num_shards, combiner, reducer):
    self.id = id
    self.shards = [Shard(i) for i in range(num_shards)]
    self.combiner = combiner
    self.reducer = reducer
  
  def update(self, shard,  k, v):
    sh = self.shards[shard]
    if k in sh.data:
      sh.data[k] = self.reducer(sh.data[k], v)
    else:
      sh.data[k] = v

  def get(shard, k):
    return self.shards[shard].data[k]


cdef char TYPE_STR = 0
cdef char TYPE_INT = 1
cdef char TYPE_FLOAT = 2
cdef char TYPE_NUMPY = 3
cdef char TYPE_OTHER = 4

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
  cdef object buf_array
  
  def __init__(self, v):
    if isinstance(v, int):
      v = N.ndarray(v, dtype=N.uint8)
    
    self.buf_array = v
    self.buffer = get_buffer(v)
    self.base = <byte*>self.buffer.buf
    self.len = self.buffer.len
    
  def __dealloc__(self):
    PyBuffer_Release(&self.buffer)
    
  cdef byte[:] view(self):
    return self.buf_array
    
  def __len__(self):
    return self.len
  
  cdef byte* data(self):
    return self.base
  
  cdef int size(self):
    return self.len


cdef class Stream:
  cdef int pos
  cdef Buffer buffer

  def __init__(self, arg=1000):
    if buffer.PyObject_CheckBuffer(arg):
      self.buffer = Buffer(arg)
    else:
      self.buffer = None
      self.reserve(arg)
    self.pos = 0

  cpdef seek(self, int pos):
    self.pos = pos
  
  cdef reserve(self, int num):
    if self.buffer is not None and self.buffer.size() - self.pos > num:
      return

    cdef int new_len = 2 * (num + self.pos)
    cdef Buffer new_buf = Buffer(new_len)
    if self.buffer is not None:
      memcpy(new_buf.data(), self.buffer.data(), self.pos)
    self.buffer = new_buf

  cpdef object getvalue(self):
    return self.buffer.view()[:self.pos]

  cdef void write_bytes(self, byte * ptr, int len):
    self.reserve(len)
    memcpy(self.buffer.data() + self.pos, ptr, len)
    # print 'Writing... %d:: %s' % (len, string.PyString_FromStringAndSize(<char*>ptr, len))
    # print 'First... %s' % (string.PyString_FromStringAndSize(<char*>self.buffer.data(), 4))
    self.pos += len

  cpdef int write(self, v):
    cdef Py_buffer data = get_buffer(v)
    self.write_bytes(< byte *> data.buf, data.len)
    PyBuffer_Release(&data)
    return self.pos

  cpdef write_str(self, str v):
    self.write_int(len(v))
    self.write_bytes(v, len(v))

  cpdef write_int(self, int v):
    self.write_bytes(< byte *> & v, 4)

  cpdef str read(self, len):
    cdef count = min(self.buffer.size() - self.pos, len)
    cdef byte* ptr = self.buffer.base + self.pos
    cdef str v = string.PyString_FromStringAndSize(<char*>ptr, count)
    self.pos += count
    #print 'Read', v
    return v

  cpdef int read_int(self):
    cdef int* v = <int*>(self.buffer.base + self.pos)
    self.pos += 4
    return v[0]

  cpdef str read_str(self):
    cdef int len = self.read_int()
    return self.read(len)
  
  cpdef int readinto(self, out):
    cdef Py_buffer b = get_buffer(out)
    cdef int count = min(self.buffer.size() - self.pos, b.len)
    memcpy(b.buf, self.buffer.data() + self.pos, count)
    self.pos += count
    return self.pos

  cpdef int size(self):
    return self.buffer.size()
  
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

Reader = Stream
Writer = Stream
