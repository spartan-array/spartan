import cPickle
from struct import unpack, pack

import numpy as np
import dis, marshal, types

from spartan.core import Message
from spartan import cloudpickle

from cpython cimport string
from libcpp.unordered_set cimport unordered_set

ctypedef unsigned char byte

cdef unordered_set[byte] GLOBAL_OPS 
GLOBAL_OPS.insert(dis.opname.index('STORE_GLOBAL'))
GLOBAL_OPS.insert(dis.opname.index('DELETE_GLOBAL'))
GLOBAL_OPS.insert(dis.opname.index('LOAD_GLOBAL'))
#cdef list GLOBAL_OPS = [dis.opname.index('STORE_GLOBAL'),
#                        dis.opname.index('DELETE_GLOBAL'),
#                        dis.opname.index('LOAD_GLOBAL')]
 
cdef byte HAVE_ARGUMENT = dis.HAVE_ARGUMENT
cdef byte EXTENDED_ARG = dis.EXTENDED_ARG

#cdef list GLOBAL_OPS = [chr(dis.opname.index('STORE_GLOBAL')),
#                        chr(dis.opname.index('DELETE_GLOBAL')),
#                        chr(dis.opname.index('LOAD_GLOBAL'))]

#cdef bytes HAVE_ARGUMENT = chr(dis.HAVE_ARGUMENT)
#cdef bytes EXTENDED_ARG = chr(dis.EXTENDED_ARG)

cdef extern from "Python.h":
  object PyCell_New(object value)

cdef void write_str(str s, f):
  write_int(len(s), f)
  f.write(s)

cdef void write_int(int i, f):
  f.write(pack('i', i))

cdef str read_str(f):
  cdef int l = read_int(f)
  return f.read(l)

cdef int read_int(f):
  return unpack('i', f.read(4))[0]

cdef set extract_code_globals(co):
  """
  Find all globals names read or written to code block co
  """
  cdef char* code = string.PyString_AsString(co.co_code)
  #cdef bytes code = co.co_code
  cdef tuple names = co.co_names
  cdef set out_names = set()
  
  cdef int n = len(co.co_code), i = 0
  cdef long oparg, extended_arg = 0
  cdef byte op
  while i < n:
    op = <byte>code[i]
  
    i += 1
    if op >= HAVE_ARGUMENT:
      oparg = code[i] + code[i+1]*256 + extended_arg
      extended_arg = 0
      i += 2
      if op == EXTENDED_ARG:
        extended_arg = oparg*65536L
      if GLOBAL_OPS.find(op) != GLOBAL_OPS.end():
        out_names.add(names[oparg])
  
  # see if nested function have any global refs
  if co.co_consts:   
    for const in co.co_consts:
      if type(const) is types.CodeType and const.co_names:
        out_names = out_names.union(extract_code_globals(const))

  return out_names

cpdef write_function(fn, f):
  cdef list closure_values = []
  cdef dict f_globals = {}
  cdef bytes var
  cdef int pos = f.tell()
  try:
    f.write('F')

    if fn.func_closure is not None:
      closure_values = [v.cell_contents for v in fn.func_closure]
   
    for var in extract_code_globals(fn.func_code):
      #Some names, such as class functions are not global - we don't need them
      val = fn.func_globals.get(var)
      if val is not None:
        f_globals[var] = val
  
    write_tuple((marshal.dumps(fn.func_code), f_globals, fn.func_name, fn.func_defaults, closure_values), f) 
  
  except Exception:
    f.seek(pos)
    f.write('P')
    cloudpickle.dump(fn, f, protocol= -1)

cpdef write_dict(dict d, f):
  f.write('D')
  write_int(len(d), f)
  for k, v in d.iteritems():
    write(k, f)
    write(v, f)
      
cpdef write_list(list l, f):
  f.write('L')
  write_int(len(l), f)
  for elem in l:
    write(elem, f)    

cpdef write_tuple(tuple t, f):
  f.write('T')
  write_int(len(t), f)
  for elem in t:
    write(elem, f)    

cpdef write_msg(m, f):
  f.write('M')
  write_str(cPickle.dumps(m.__class__), f)
  write_dict(m.__dict__, f)

cpdef write_numpy(n, f):
  if not np.ma.isMaskedArray(n):
    # sparse arrays will fail the np.ndarray check and are handled by cPickle
    f.write('N')
    if not n.flags['C_CONTIGUOUS']:
      n = np.ascontiguousarray(n)
    cPickle.dump(n.dtype, f, protocol= -1)
    cPickle.dump(n.shape, f, protocol= -1)
    cPickle.dump(n.strides, f, protocol= -1)
    write_int(n.nbytes, f)
    f.write(n)
  else:
    f.write('P')
    cPickle.dump(n, f, protocol= -1)
    
cpdef write_pickle(p, f):
  f.write('P')
  try:
    #print 'Using cpickle for ', obj
    v = cPickle.dumps(p, -1)
    f.write(v)
  except (TypeError, cPickle.PicklingError, cPickle.PickleError):
    # print 'Using cloudpickle for ', obj
    cloudpickle.dump(p, f, protocol= -1)

cdef dict type_to_writer = {types.DictType : write_dict,
                            types.ListType : write_list,
                            types.TupleType : write_tuple,
                            types.FunctionType : write_function,
                            types.NoneType : lambda obj, f: f.write('E'),
                            np.core.numerictypes.ndarray : write_numpy,
                           }
cpdef write(obj, f):
  t = type(obj)
  write_fn = type_to_writer.get(t)
  if write_fn == None:
    if t.__base__ == Message:
      write_msg(obj, f)
    else:
      write_pickle(obj, f)  
  else:
    write_fn(obj, f)

cdef dict type_to_reader = {'D' : read_dict,
                            'L' : read_list,
                            'T' : read_tuple,
                            'F' : read_function,
                            'E' : lambda f: None,
                            'N' : read_numpy,
                            'M' : read_msg,
                            'P' : cPickle.load,
                           }

cpdef read_function(f):
  f.read(1)
  code_str, globals, name, defaults, closure_values = read_tuple(f)
  
  globals['__builtins__'] = __builtins__
  closure = tuple([PyCell_New(v) for v in closure_values])
  
  return types.FunctionType(marshal.loads(code_str), globals, name, defaults, closure)

cpdef read_numpy(f):
  dtype = cPickle.load(f)
  shape = cPickle.load(f)
  strides = cPickle.load(f)
  b = read_str(f)
  return np.ndarray(shape, buffer=b, strides=strides, dtype=dtype)

cpdef read_msg(f):
  klass = cPickle.loads(read_str(f))
  f.read(1)
  args = read_dict(f)
  return klass(**args)

cpdef read_tuple(f):
  cdef int sz = read_int(f)
  return tuple([read(f) for i in range(sz)])

cpdef read_list(f):
  cdef int sz = read_int(f)
  return [read(f) for i in range(sz)]

cpdef read_dict(f):
  cdef int i, sz = read_int(f)
  cdef dict d = {}
  for i in range(sz):
    k = read(f)
    d[k] = read(f)
  return d

cpdef read(f):
  datatype = f.read(1)
  reader = type_to_reader.get(datatype)
  if reader == None:
    raise KeyError, 'Unknown datatype: "%s"' % datatype
  return reader(f)

