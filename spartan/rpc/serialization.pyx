import cPickle
from struct import unpack, pack

import numpy as np
import dis, marshal, types

from ..core import Message
from .. import cloudpickle
import time
from spartan import util

from libc.string cimport memcpy

cimport numpy as N
import numpy as N

from cpython cimport buffer
from cpython cimport string

from cython.view cimport memoryview

ctypedef unsigned char byte

cdef list GLOBAL_OPS = [chr(dis.opname.index('STORE_GLOBAL')),
                        chr(dis.opname.index('DELETE_GLOBAL')),
                        chr(dis.opname.index('LOAD_GLOBAL'))]

cdef bytes HAVE_ARGUMENT = chr(dis.HAVE_ARGUMENT)
cdef bytes EXTENDED_ARG = chr(dis.EXTENDED_ARG)

cdef extern from "Python.h":
  object PyCell_New(object value)
  object PyMemoryView_FromBuffer(Py_buffer *)
  object PyMemoryView_FromObject(object)
  object PyBuffer_FromMemory(void*, Py_ssize_t)
  object PyBuffer_FromReadWriteMemory(void*, Py_ssize_t)
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

cpdef write(obj, f):
  _write(obj, f)

cpdef write_str(s, f):
  write_int(len(s), f)
  f.write(s)

cpdef write_int(i, f):
  f.write(pack('i', i))

cpdef read_str(f):
  l = read_int(f)
  return f.read(l)

cpdef read_int(f):
  b = f.read(4)
  return unpack('i', b)[0]

cpdef extract_code_globals(co):
  """
  Find all globals names read or written to code block co
  """
  cdef bytes code = co.co_code
  cdef tuple names = co.co_names
  cdef set out_names = set()
  
  cdef int n = len(code), i = 0
  cdef long oparg, extended_arg = 0
  cdef bytes op
  while i < n:
    op = code[i]
  
    i = i+1
    if op >= HAVE_ARGUMENT:
      oparg = ord(code[i]) + ord(code[i+1])*256 + extended_arg
      extended_arg = 0
      i = i+2
      if op == EXTENDED_ARG:
        extended_arg = oparg*65536L
      if op in GLOBAL_OPS:
        out_names.add(names[oparg])
  
  # see if nested function have any global refs
  if co.co_consts:   
    for const in co.co_consts:
      if type(const) is types.CodeType and const.co_names:
        out_names = out_names.union(extract_code_globals(const))

  return out_names

cpdef write_function(fn, f):
  cdef list closure_values = []
  if fn.func_closure is not None:
    closure_values = [v.cell_contents for v in fn.func_closure]
    
  cdef set func_global_refs = extract_code_globals(fn.func_code)
                    
  cdef dict f_globals = {}
  for var in func_global_refs:
    #Some names, such as class functions are not global - we don't need them
    if fn.func_globals.has_key(var):
      f_globals[var] = fn.func_globals[var]

  _write((marshal.dumps(fn.func_code), f_globals, fn.func_name, fn.func_defaults, closure_values), f)

cpdef read_function(f):
  code_str, globals, name, defaults, closure_values = read(f)
  
  globals['__builtins__'] = __builtins__
  closure = tuple([PyCell_New(v) for v in closure_values])
  
  return types.FunctionType(marshal.loads(code_str), globals, name, defaults, closure)

cpdef _write(obj, f):
  if obj is None:
    f.write('E')
  elif isinstance(obj, np.ndarray) and not np.ma.isMaskedArray(obj):
    # sparse arrays will fail the np.ndarray check and are handled by cPickle
    f.write('N')
    if not obj.flags['C_CONTIGUOUS']:
      obj = np.ascontiguousarray(obj)
    cPickle.dump(obj.dtype, f, protocol= -1)
    cPickle.dump(obj.shape, f, protocol= -1)
    cPickle.dump(obj.strides, f, protocol= -1)
    write_int(obj.nbytes, f)
    f.write(obj)
    #write_str(buffer(obj), f)
  elif isinstance(obj, Message):
    f.write('M')
    write_str(cPickle.dumps(obj.__class__), f)
    _write(obj.__dict__, f)
  elif isinstance(obj, tuple):
    f.write('T')
    write_int(len(obj), f)
    for elem in obj:
      _write(elem, f)
  elif isinstance(obj, list):
    f.write('L')
    write_int(len(obj), f)
    for elem in obj:
      _write(elem, f)
  elif isinstance(obj, dict):
    f.write('D')
    write_int(len(obj), f)
    for k, v in obj.iteritems():
      _write(k, f)
      _write(v, f)
  elif type(obj) == types.FunctionType:
    pos = f.tell()
    try:
      f.write('F')
      write_function(obj, f)
    except Exception:
      f.seek(pos)
      f.write('P')
      cloudpickle.dump(obj, f, protocol= -1)
  else:
    f.write('P')
    try:
      # print 'Using cpickle for ', obj
      v = cPickle.dumps(obj, -1)
      f.write(v)
    except (TypeError, cPickle.PicklingError, cPickle.PickleError):
      # print 'Using cloudpickle for ', obj
      cloudpickle.dump(obj, f, protocol= -1)

cpdef read(f):
  datatype = f.read(1)
  if datatype == 'E':
    return None
  elif datatype == 'N':
    dtype = cPickle.load(f)
    shape = cPickle.load(f)
    strides = cPickle.load(f)
    b = read_str(f)
    array = np.ndarray(shape, buffer=b, strides=strides, dtype=dtype)
    return array
  elif datatype == 'M':
    klass = cPickle.loads(read_str(f))
    args = read(f)
    return klass(**args)
  elif datatype == 'T':
    sz = read_int(f)
    return tuple([read(f) for i in range(sz)])
  elif datatype == 'L':
    sz = read_int(f)
    return [read(f) for i in range(sz)]
  elif datatype == 'D':
    sz = read_int(f)
    lst = []
    for i in range(sz):
      k = read(f)
      v = read(f)
      lst.append((k, v))
    return dict(lst)
  elif datatype == 'F':
    return read_function(f)
  elif datatype == 'P':
    res = cPickle.load(f)
    return res

  raise KeyError, 'Unknown datatype: "%s"' % datatype
