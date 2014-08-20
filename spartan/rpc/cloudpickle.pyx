# distutils: language = c++

"""
This module was extracted from the `cloud` package, developed by `PiCloud, Inc.
<http://www.picloud.com>`_.  

(This version comes from PySpark).

This class is defined to override standard pickle functionality

The goals of it follow:
-Serialize lambdas and nested functions to compiled byte code
-Deal with main module correctly
-Deal with other non-serializable objects

It does not include an unpickler, as standard python unpickling suffices.

"""

# Copyright (c) 2012, Regents of the University of California.
# Copyright (c) 2009 `PiCloud, Inc. <http://www.picloud.com>`_.
# All rights reserved.
# 
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#     * Redistributions of source code must retain the above copyright
#       notice, this list of conditions and the following disclaimer.
#     * Redistributions in binary form must reproduce the above copyright
#       notice, this list of conditions and the following disclaimer in the
#       documentation and/or other materials provided with the distribution.
#     * Neither the name of the University of California, Berkeley nor the
#       names of its contributors may be used to endorse or promote
#       products derived from this software without specific prior written
#       permission.
# 
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
# "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
# LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
# A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
# HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
# SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED
# TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
# PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
# LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
# NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
# SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.


import operator
import os
import struct
import pickle
import sys
from types import *
from functools import partial
import itertools
from copy_reg import dispatch_table, _extension_registry, _inverted_registry, _extension_cache
import new
import dis
import traceback
import re
cimport cython
from cpython cimport bool
from .serialization_buffer import Writer

# Keep in synch with cPickle.  This is the highest protocol number we
# know how to read.
cdef int HIGHEST_PROTOCOL = 2

# Pickle opcodes.  See pickletools.py for extensive docs.  The listing
# here is in kind-of alphabetical order of 1-character pickle code.
# pickletools groups them by purpose.

cdef str MARK            = '('   # push special markobject on stack
cdef str STOP            = '.'   # every pickle ends with STOP
cdef str POP             = '0'   # discard topmost stack item
cdef str POP_MARK        = '1'   # discard stack top through topmost markobject
cdef str DUP             = '2'   # duplicate top stack item
cdef str FLOAT           = 'F'   # push float object; decimal string argument
cdef str INT             = 'I'   # push integer or bool; decimal string argument
cdef str BININT          = 'J'   # push four-byte signed int
cdef str BININT1         = 'K'   # push 1-byte unsigned int
cdef str LONG            = 'L'   # push long; decimal string argument
cdef str BININT2         = 'M'   # push 2-byte unsigned int
cdef str NONE            = 'N'   # push None
cdef str PERSID          = 'P'   # push persistent object; id is taken from string arg
cdef str BINPERSID       = 'Q'   #  "       "         "  ;  "  "   "     "  stack
cdef str REDUCE          = 'R'   # apply callable to argtuple, both on stack
cdef str STRING          = 'S'   # push string; NL-terminated string argument
cdef str BINSTRING       = 'T'   # push string; counted binary string argument
cdef str SHORT_BINSTRING = 'U'   #  "     "   ;    "      "       "      " < 256 bytes
cdef str UNICODE         = 'V'   # push Unicode string; raw-unicode-escaped'd argument
cdef str BINUNICODE      = 'X'   #   "     "       "  ; counted UTF-8 string argument
cdef str APPEND          = 'a'   # append stack top to list below it
cdef str BUILD           = 'b'   # call __setstate__ or __dict__.update()
cdef str GLOBAL          = 'c'   # push self.find_class(modname, name); 2 string args
cdef str DICT            = 'd'   # build a dict from stack items
cdef str EMPTY_DICT      = '}'   # push empty dict
cdef str APPENDS         = 'e'   # extend list on stack by topmost stack slice
cdef str GET             = 'g'   # push item from memo on stack; index is string arg
cdef str BINGET          = 'h'   #   "    "    "    "   "   "  ;   "    " 1-byte arg
cdef str INST            = 'i'   # build & push class instance
cdef str LONG_BINGET     = 'j'   # push item from memo on stack; index is 4-byte arg
cdef str LIST            = 'l'   # build list from topmost stack items
cdef str EMPTY_LIST      = ']'   # push empty list
cdef str OBJ             = 'o'   # build & push class instance
cdef str PUT             = 'p'   # store stack top in memo; index is string arg
cdef str BINPUT          = 'q'   #   "     "    "   "   " ;   "    " 1-byte arg
cdef str LONG_BINPUT     = 'r'   #   "     "    "   "   " ;   "    " 4-byte arg
cdef str SETITEM         = 's'   # add key+value pair to dict
cdef str TUPLE           = 't'   # build tuple from topmost stack items
cdef str EMPTY_TUPLE     = ')'   # push empty tuple
cdef str SETITEMS        = 'u'   # modify dict by adding topmost key+value pairs
cdef str BINFLOAT        = 'G'   # push float; arg is 8-byte float encoding

cdef str TRUE            = 'I01\n'  # not an opcode; see INT docs in pickletools.py
cdef str FALSE           = 'I00\n'  # not an opcode; see INT docs in pickletools.py

# Protocol 2

cdef str PROTO           = '\x80'  # identify pickle protocol
cdef str NEWOBJ          = '\x81'  # build object by applying cls.__new__ to argtuple
cdef str EXT1            = '\x82'  # push object from extension registry; 1-byte index
cdef str EXT2            = '\x83'  # ditto, but 2-byte index
cdef str EXT4            = '\x84'  # ditto, but 4-byte index
cdef str TUPLE1          = '\x85'  # build 1-tuple from stack top
cdef str TUPLE2          = '\x86'  # build 2-tuple from two topmost stack items
cdef str TUPLE3          = '\x87'  # build 3-tuple from three topmost stack items
cdef str NEWTRUE         = '\x88'  # push True
cdef str NEWFALSE        = '\x89'  # push False
cdef str LONG1           = '\x8a'  # push long from < 256 bytes
cdef str LONG4           = '\x8b'  # push really big long

cdef list _tuplesize2code = [EMPTY_TUPLE, TUPLE1, TUPLE2, TUPLE3]

#relevant opcodes
cdef bytes STORE_GLOBAL = chr(dis.opname.index('STORE_GLOBAL'))
cdef bytes DELETE_GLOBAL = chr(dis.opname.index('DELETE_GLOBAL'))
cdef bytes LOAD_GLOBAL = chr(dis.opname.index('LOAD_GLOBAL'))
cdef list GLOBAL_OPS = [STORE_GLOBAL, DELETE_GLOBAL, LOAD_GLOBAL]

cdef bytes HAVE_ARGUMENT = chr(dis.HAVE_ARGUMENT)
cdef bytes EXTENDED_ARG = chr(dis.EXTENDED_ARG)

import logging
cloudLog = logging.getLogger("Cloud.Transport")

# Jython has PyStringMap; it's a dict subclass with string keys
try:
    from org.python.core import PyStringMap
except ImportError:
    PyStringMap = None

# UnicodeType may or may not be exported (normally imported from types)
try:
    UnicodeType
except NameError:
    UnicodeType = None
    
cdef list PyObject_HEAD
try:
    import ctypes
except (MemoryError, ImportError):
    logging.warning('Exception raised on importing ctypes. Likely python bug.. some functionality will be disabled', exc_info = True)
    ctypes = None
    PyObject_HEAD = None
else:

    # for reading internal structures
    PyObject_HEAD = [
        ('ob_refcnt', ctypes.c_size_t),
        ('ob_type', ctypes.c_void_p),
    ]


try:
    from cStringIO import StringIO
except ImportError:
    from StringIO import StringIO

# These helper functions were copied from PiCloud's util module.
cdef bool islambda(func):
    return getattr(func,'func_name') == '<lambda>'

cdef tuple xrange_params(xrangeobj):
    """Returns a 3 element tuple describing the xrange start, step, and len
    respectively

    Note: Only guarentees that elements of xrange are the same. parameters may
    be different.
    e.g. xrange(1,1) is interpretted as xrange(0,0); both behave the same
    though w/ iteration
    """

    cdef int xrange_len = len(xrangeobj)
    if not xrange_len: #empty
        return (0,1,0)
    
    cdef int start = xrangeobj[0]
    if xrange_len == 1: #one element
        return (start, 1, 1)
    return (start, xrangeobj[1] - xrangeobj[0], xrange_len)

# This is true for Jython
cpdef save_string(CloudPickler self, str obj, pack=struct.pack):
    cdef bool unicode = obj.isunicode()
    cdef int l
    
    if self.bin:
        if unicode:
            obj = obj.encode("utf-8")
        l = len(obj)
        if l < 256 and not unicode:
            self.write(SHORT_BINSTRING + chr(l) + obj)
        else:
            s = pack("<i", l)
            if unicode:
                self.write(BINUNICODE + s + obj)
            else:
                self.write(BINSTRING + s + obj)
    else:
        if unicode:
            obj = obj.replace("\\", "\\u005c")
            obj = obj.replace("\n", "\\u000a")
            obj = obj.encode('raw-unicode-escape')
            self.write(UNICODE + obj + '\n')
        else:
            self.write(STRING + repr(obj) + '\n')
    self.memoize(obj)
        
#debug variables intended for developer use:
cdef bool printSerialization = False
cdef bool printMemoization = False

cdef bool useForcedImports = True #Should I use forced imports for tracking?

cdef dict dispatch = {buffer              : CloudPickler.save_buffer,
                      GeneratorType       : CloudPickler.save_unsupported,
                      ModuleType          : CloudPickler.save_module,
                      CodeType            : CloudPickler.save_codeobject,
                      FunctionType        : CloudPickler.save_function,
                      BuiltinFunctionType : CloudPickler.save_global,
                      TypeType            : CloudPickler.save_global,
                      ClassType           : CloudPickler.save_global,
                      MethodType          : CloudPickler.save_instancemethod,
                      property            : CloudPickler.save_property,
                      NoneType            : CloudPickler.save_none,
                      bool                : CloudPickler.save_bool,
                      IntType             : CloudPickler.save_int,
                      LongType            : CloudPickler.save_long,
                      FloatType           : CloudPickler.save_float,
                      StringType          : CloudPickler.save_string,
                      UnicodeType         : CloudPickler.save_unicode,
                      TupleType           : CloudPickler.save_tuple,
                      ListType            : CloudPickler.save_list,
                      DictionaryType      : CloudPickler.save_dict,
                      InstanceType        : CloudPickler.save_inst,
                      file                : CloudPickler.save_file,                 
                      }

if StringType is UnicodeType:
    dispatch[StringType] = save_string
    
if PyObject_HEAD:
    dispatch[operator.itemgetter] = CloudPickler.save_itemgetter  

if PyStringMap:
    dispatch[PyStringMap] = CloudPicker.save_dict 
    
#python2.6+ supports slice pickling. some py2.5 extensions might as well.  We just test it
try:
    slice(0,1).__reduce__()
except TypeError: #can't pickle -
    dispatch[slice] = CloudPickler.save_unsupported

#python2.6+ supports xrange pickling. some py2.5 extensions might as well.  We just test it
try:
    xrange(0).__reduce__()
except TypeError: #can't pickle -- use PiCloud pickler
    dispatch[xrange] = CloudPickler.save_xrange

if sys.version_info < (2,7): #2.7 supports partial pickling
    dispatch[partial] = CloudPickler.save_partial
    
#itertools objects do not pickle!
for v in itertools.__dict__.values():
    if type(v) is type:
        dispatch[v] = CloudPickler.save_unsupported
             
cdef class CloudPickler:
    cdef dict memo, globals_ref   # map ids to dictionary. used to ensure that functions can share global env
    cdef bool bin, savedForceImports, savedDjangoEnv  #hack tro transport django environment
    cdef set modules        #set of modules needed to depickle
    cdef int proto, fast, _BATCHSIZE
    cdef list numpy_tst_mods
    cdef object write
    
    def __init__(self, file, protocol=0, min_size_to_save=0):
        self.proto = <int>protocol
        if self.proto < 0:
            self.proto = HIGHEST_PROTOCOL
        elif self.proto > HIGHEST_PROTOCOL:
            raise ValueError("pickle protocol must be <= %d" % HIGHEST_PROTOCOL)
          
        self.bin = self.proto >= 1
        self.write = file.write
        self.memo = {}
        
        self.fast = 0

        self.savedForceImports = False
        self.savedDjangoEnv = False
        self.modules = set()
        self.globals_ref = {}
        self._BATCHSIZE = 1000
        
        self.numpy_tst_mods = ['numpy', 'scipy.special']
    
    cdef void clear_memo(self):
        self.memo.clear()
          
    cdef void dump(self, obj):
        # note: not thread safe
        # minimal side-effects, so not fixing
        cdef int recurse_limit = 3000
        cdef int base_recurse = sys.getrecursionlimit()
        cdef int new_recurse
        if base_recurse < recurse_limit:
            sys.setrecursionlimit(recurse_limit)
        self.inject_addons()
        try:
            if self.proto >= 2:
                self.write(PROTO + chr(self.proto))
            self.save(obj)
            self.write(STOP)
        except RuntimeError, e:
            if 'recursion' in e.args[0]:
                raise pickle.PicklingError("""Could not pickle object as excessively deep recursion required.
                    Try _fast_serialization=2 or contact PiCloud support""")
        finally:
            new_recurse = sys.getrecursionlimit()
            if new_recurse == recurse_limit:
                sys.setrecursionlimit(base_recurse)

    # Return a PUT (BINPUT, LONG_BINPUT) opcode string, with argument i.
    cdef str put(self, int i, pack=struct.pack):
        if self.bin:
            if i < 256:
                return BINPUT + chr(i)
            else:
                return LONG_BINPUT + pack("<i", i)

        return PUT + repr(i) + '\n'

    # Return a GET (BINGET, LONG_BINGET) opcode string, with argument i.
    cdef str get(self, int i, pack=struct.pack):
        if self.bin:
            if i < 256:
                return BINGET + chr(i)
            else:
                return LONG_BINGET + pack("<i", i)

        return GET + repr(i) + '\n'
    
    def save(self, obj):
        # Check the memo
        x = self.memo.get(id(obj))
        if x:
            self.write(self.get(x[0]))
            return

        # Check the type dispatch table
        t = type(obj)
        f = dispatch.get(t)
        if f:
            f(self, obj) # Call unbound method with explicit self
            return

        # Check copy_reg.dispatch_table
        reduce = dispatch_table.get(t)
        if reduce:
            rv = reduce(obj)
        else:
            # Check for a class with a custom metaclass; treat as regular class
            try:
                issc = issubclass(t, TypeType)
            except TypeError: # t is not a class (old Boost; see SF #502085)
                issc = 0
            if issc:
                self.save_global(obj)
                return

            # Check for a __reduce_ex__ method, fall back to __reduce__
            reduce = getattr(obj, "__reduce_ex__", None)
            if reduce:
                rv = reduce(self.proto)
            else:
                reduce = getattr(obj, "__reduce__", None)
                if reduce:
                    rv = reduce()
                else:
                    raise pickle.PicklingError("Can't pickle %r object: %r" %
                                        (t.__name__, obj))

        # Check for string returned by reduce(), meaning "save as global"
        if type(rv) is StringType:
            self.save_global(obj, rv)
            return

        # Assert that reduce() returned a tuple
        if type(rv) is not TupleType:
            raise pickle.PicklingError("%s must return string or tuple" % reduce)

        # Assert that it returned an appropriately sized tuple
        l = len(rv)
        if not (2 <= l <= 5):
            raise pickle.PicklingError("Tuple returned by %s must have "
                                "two to five elements" % reduce)

        # Save the reduce() output and finally memoize the object
        self.save_reduce(obj=obj, *rv)

    def save_pers(self, pid):
        # Save a persistent id reference
        if self.bin:
            self.save(pid)
            self.write(BINPERSID)
        else:
            self.write(PERSID + str(pid) + '\n')

    def save_buffer(self, obj):
        """Fallback to save_string"""
        self.save_string(str(obj))

    #block broken objects
    def save_unsupported(self, obj, pack=None):
        raise pickle.PicklingError("Cannot pickle objects of type %s" % type(obj))

    def save_module(self, obj, pack=struct.pack):
        """
        Save a module as an import
        """
        #print 'try save import', obj.__name__
        self.modules.add(obj)
        self.save_reduce(subimport,(obj.__name__,), obj=obj)

    def save_codeobject(self, obj, pack=struct.pack):
        """
        Save a code object
        """
        #print 'try to save codeobj: ', obj
        args = (
            obj.co_argcount, obj.co_nlocals, obj.co_stacksize, obj.co_flags, obj.co_code,
            obj.co_consts, obj.co_names, obj.co_varnames, obj.co_filename, obj.co_name,
            obj.co_firstlineno, obj.co_lnotab, obj.co_freevars, obj.co_cellvars
        )
        self.save_reduce(CodeType, args, obj=obj)

    def save_function(self, obj, name=None, pack=struct.pack):
        """ Registered with the dispatch to handle all function types.

        Determines what kind of function obj is (e.g. lambda, defined at
        interactive prompt, etc) and handles the pickling appropriately.
        """
        write = self.write

        name = obj.__name__
        modname = whichmodule(obj, name)
        #print 'which gives %s %s %s' % (modname, obj, name)
        try:
            themodule = sys.modules[modname]
        except KeyError: # eval'd items such as namedtuple give invalid items for their function __module__
            modname = '__main__'

        if modname == '__main__':
            themodule = None

        if themodule:
            self.modules.add(themodule)

        if not self.savedDjangoEnv:
            #hack for django - if we detect the settings module, we transport it
            django_settings = os.environ.get('DJANGO_SETTINGS_MODULE', '')
            if django_settings:
                django_mod = sys.modules.get(django_settings)
                if django_mod:
                    cloudLog.debug('Transporting django settings %s during save of %s', django_mod, name)
                    self.savedDjangoEnv = 1
                    self.modules.add(django_mod)
                    write(MARK)
                    self.save_reduce(django_settings_load, (django_mod.__name__,), obj=django_mod)
                    write(POP_MARK)


        # if func is lambda, def'ed at prompt, is in main, or is nested, then
        # we'll pickle the actual function object rather than simply saving a
        # reference (as is done in default pickler), via save_function_tuple.
        if islambda(obj) or obj.func_code.co_filename == '<stdin>' or themodule == None:
            #Force server to import modules that have been imported in main
            modList = None
            if themodule == None and not self.savedForceImports:
                mainmod = sys.modules['__main__']
                if useForcedImports and hasattr(mainmod,'___pyc_forcedImports__'):
                    modList = list(mainmod.___pyc_forcedImports__)
                self.savedForceImports = True
            self.save_function_tuple(obj, modList)
            return
        else:   # func is nested
            klass = getattr(themodule, name, None)
            if klass is None or klass is not obj:
                self.save_function_tuple(obj, [themodule])
                return

        if obj.__dict__:
            # essentially save_reduce, but workaround needed to avoid recursion
            self.save(_restore_attr)
            write(MARK + GLOBAL + modname + '\n' + name + '\n')
            self.memoize(obj)
            self.save(obj.__dict__)
            write(TUPLE + REDUCE)
        else:
            write(GLOBAL + modname + '\n' + name + '\n')
            self.memoize(obj)

    def save_function_tuple(self, func, forced_imports):
        """  Pickles an actual func object.

        A func comprises: code, globals, defaults, closure, and dict.  We
        extract and save these, injecting reducing functions at certain points
        to recreate the func object.  Keep in mind that some of these pieces
        can contain a ref to the func itself.  Thus, a naive save on these
        pieces could trigger an infinite loop of save's.  To get around that,
        we first create a skeleton func object using just the code (this is
        safe, since this won't contain a ref to the func), and memoize it as
        soon as it's created.  The other stuff can then be filled in later.
        """

        # save the modules (if any)
        if forced_imports:
            self.write(MARK)
            self.save(_modules_to_main)
            #print 'forced imports are', forced_imports

            forced_names = map(lambda m: m.__name__, forced_imports)
            self.save((forced_names,))

            #save((forced_imports,))
            self.write(REDUCE)
            self.write(POP_MARK)

        code, f_globals, defaults, closure, dct, base_globals = self.extract_func_data(func)

        self.save(_fill_function)  # skeleton function updater
        self.write(MARK)    # beginning of tuple that _fill_function expects

        # create a skeleton function object and memoize it
        self.save(_make_skel_func)
        self.save((code, len(closure), base_globals))
        self.write(REDUCE)
        self.memoize(func)

        # save the rest of the func data needed by _fill_function
        self.save(f_globals)
        self.save(defaults)
        self.save(closure)
        self.save(dct)
        self.write(TUPLE)
        self.write(REDUCE)  # applies _fill_function on the tuple

    cdef set extract_code_globals(self, co):
        """
        Find all globals names read or written to by codeblock co
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
        #print 'extracted', out_names, ' from ', names
        return out_names

    def extract_func_data(self, func):
        """
        Turn the function into a tuple of data necessary to recreate it:
            code, globals, defaults, closure, dict
        """
        code = func.func_code

        # extract all global ref's
        func_global_refs = self.extract_code_globals(code)
        if code.co_consts:   # see if nested function have any global refs
            for const in code.co_consts:
                if type(const) is CodeType and const.co_names:
                    func_global_refs = func_global_refs.union( self.extract_code_globals(const))
        # process all variables referenced by global environment
        f_globals = {}
        for var in func_global_refs:
            #Some names, such as class functions are not global - we don't need them
            if func.func_globals.has_key(var):
                f_globals[var] = func.func_globals[var]

        # defaults requires no processing
        defaults = func.func_defaults

        def get_contents(cell):
            try:
                return cell.cell_contents
            except ValueError, e: #cell is empty error on not yet assigned
                raise pickle.PicklingError('Function to be pickled has free variables that are referenced before assignment in enclosing scope')

        closure = []
        # process closure
        if func.func_closure:
            closure = map(get_contents, func.func_closure)

        # save the dict
        dct = func.func_dict
        if printSerialization:
            outvars = ['code: ' + str(code) ]
            outvars.append('globals: ' + str(f_globals))
            outvars.append('defaults: ' + str(defaults))
            outvars.append('closure: ' + str(closure))
            print 'function ', func, 'is extracted to: ', ', '.join(outvars)

        base_globals = self.globals_ref.get(id(func.func_globals), {})
        self.globals_ref[id(func.func_globals)] = base_globals

        return (code, f_globals, defaults, closure, dct, base_globals)

    def save_global(self, obj, name=None, pack=struct.pack):
        if name is None:
            name = obj.__name__

        modname = getattr(obj, "__module__", None)
        if modname is None:
            modname = whichmodule(obj, name)

        try:
            __import__(modname)
            themodule = sys.modules[modname]
        except (ImportError, KeyError, AttributeError):  #should never occur
            raise pickle.PicklingError(
                "Can't pickle %r: Module %s cannot be found" %
                (obj, modname))

        if modname == '__main__':
            themodule = None

        if themodule:
            self.modules.add(themodule)

        sendRef = True
        typ = type(obj)
        #print 'saving', obj, typ
        try:
            try: #Deal with case when getattribute fails with exceptions
                klass = getattr(themodule, name)
            except (AttributeError):
                if modname == '__builtin__':  #new.* are misrepeported
                    modname = 'new'
                    __import__(modname)
                    themodule = sys.modules[modname]
                    try:
                        klass = getattr(themodule, name)
                    except AttributeError, a:
                        #print themodule, name, obj, type(obj)
                        raise pickle.PicklingError("Can't pickle builtin %s" % obj)
                else:
                    raise

        except (ImportError, KeyError, AttributeError):
            if typ == TypeType or typ == ClassType:
                sendRef = False
            else: #we can't deal with this
                raise
        else:
            if klass is not obj and (typ == TypeType or typ == ClassType):
                sendRef = False

        if not sendRef:
            #note: Third party types might crash this - add better checks!
            d = dict(obj.__dict__) #copy dict proxy to a dict
            if not isinstance(d.get('__dict__', None), property): # don't extract dict that are properties
                d.pop('__dict__',None)
            d.pop('__weakref__',None)

            # hack as __new__ is stored differently in the __dict__
            new_override = d.get('__new__', None)
            if new_override:
                d['__new__'] = obj.__new__

            self.save_reduce(type(obj),(obj.__name__,obj.__bases__,
                                   d),obj=obj)
            #print 'internal reduce dask %s %s'  % (obj, d)
            return

        if self.proto >= 2:
            code = _extension_registry.get((modname, name))
            if code:
                assert code > 0
                if code <= 0xff:
                    self.write(EXT1 + chr(code))
                elif code <= 0xffff:
                    self.write("%c%c%c" % (EXT2, code&0xff, code>>8))
                else:
                    self.write(EXT4 + pack("<i", code))
                return

        self.write(GLOBAL + modname + '\n' + name + '\n')
        self.memoize(obj)
      
    def save_instancemethod(self, obj):
        #Memoization rarely is ever useful due to python bounding
        self.save_reduce(MethodType, (obj.im_func, obj.im_self,obj.im_class), obj=obj)

    def save_inst_logic(self, obj):
        """Inner logic to save instance. Based off pickle.save_inst
        Supports __transient__"""
        cls = obj.__class__

        if hasattr(obj, '__getinitargs__'):
            args = obj.__getinitargs__()
            len(args) # XXX Assert it's a sequence
            _keep_alive(args, self.memo)
        else:
            args = ()

        self.write(MARK)

        if self.bin:
            self.save(cls)
            for arg in args:
                self.save(arg)
            self.write(OBJ)
        else:
            for arg in args:
                self.save(arg)
            self.write(INST + cls.__module__ + '\n' + cls.__name__ + '\n')

        self.memoize(obj)

        try:
            getstate = obj.__getstate__
        except AttributeError:
            stuff = obj.__dict__
            #remove items if transient
            if hasattr(obj, '__transient__'):
                transient = obj.__transient__
                stuff = stuff.copy()
                for k in list(stuff.keys()):
                    if k in transient:
                        del stuff[k]
        else:
            stuff = getstate()
            _keep_alive(stuff, self.memo)
        self.save(stuff)
        self.write(BUILD)


    def save_inst(self, obj):
        # Hack to detect PIL Image instances without importing Imaging
        # PIL can be loaded with multiple names, so we don't check sys.modules for it
        if hasattr(obj,'im') and hasattr(obj,'palette') and 'Image' in obj.__module__:
            self.save_image(obj)
        else:
            self.save_inst_logic(obj)

    def save_property(self, obj):
        # properties not correctly saved in python
        self.save_reduce(property, (obj.fget, obj.fset, obj.fdel, obj.__doc__), obj=obj)

    def save_itemgetter(self, obj):
        """itemgetter serializer (needed for namedtuple support)
        a bit of a pain as we need to read ctypes internals"""
        class ItemGetterType(ctypes.Structure):
            _fields_ = PyObject_HEAD + [
                ('nitems', ctypes.c_size_t),
                ('item', ctypes.py_object)
            ]


        itemgetter_obj = ctypes.cast(ctypes.c_void_p(id(obj)), ctypes.POINTER(ItemGetterType)).contents
        return self.save_reduce(operator.itemgetter, (itemgetter_obj.item,))

    def save_none(self, obj):
        self.write(NONE)

    def save_bool(self, obj):
        if self.proto >= 2:
            self.write(obj and NEWTRUE or NEWFALSE)
        else:
            self.write(obj and TRUE or FALSE)

    def save_int(self, obj, pack=struct.pack):
        if self.bin:
            # If the int is small enough to fit in a signed 4-byte 2's-comp
            # format, we can store it more efficiently than the general
            # case.
            # First one- and two-byte unsigned ints:
            if obj >= 0:
                if obj <= 0xff:
                    self.write(BININT1 + chr(obj))
                    return
                if obj <= 0xffff:
                    self.write("%c%c%c" % (BININT2, obj&0xff, obj>>8))
                    return
            # Next check for 4-byte signed ints:
            high_bits = obj >> 31  # note that Python shift sign-extends
            if high_bits == 0 or high_bits == -1:
                # All high bits are copies of bit 2**31, so the value
                # fits in a 4-byte signed int.
                self.write(BININT + pack("<i", obj))
                return
        # Text pickle, or int too big to fit in signed 4-byte format.
        self.write(INT + repr(obj) + '\n')

    def save_long(self, obj, pack=struct.pack):
        if self.proto >= 2:
            bytes = encode_long(obj)
            n = len(bytes)
            if n < 256:
                self.write(LONG1 + chr(n) + bytes)
            else:
                self.write(LONG4 + pack("<i", n) + bytes)
            return
        self.write(LONG + repr(obj) + '\n')

    def save_float(self, obj, pack=struct.pack):
        if self.bin:
            self.write(BINFLOAT + pack('>d', obj))
        else:
            self.write(FLOAT + repr(obj) + '\n')

    def save_string(self, obj, pack=struct.pack):
        if self.bin:
            n = len(obj)
            if n < 256:
                self.write(SHORT_BINSTRING + chr(n) + obj)
            else:
                self.write(BINSTRING + pack("<i", n) + obj)
        else:
            self.write(STRING + repr(obj) + '\n')
        self.memoize(obj)

    def save_unicode(self, obj, pack=struct.pack):
        if self.bin:
            encoding = obj.encode('utf-8')
            n = len(encoding)
            self.write(BINUNICODE + pack("<i", n) + encoding)
        else:
            obj = obj.replace("\\", "\\u005c")
            obj = obj.replace("\n", "\\u000a")
            self.write(UNICODE + obj.encode('raw-unicode-escape') + '\n')
        self.memoize(obj)

    def save_tuple(self, obj):
        n = len(obj)
        if n == 0:
            if self.proto:
                self.write(EMPTY_TUPLE)
            else:
                self.write(MARK + TUPLE)
            return

        if n <= 3 and self.proto >= 2:
            for element in obj:
                self.save(element)
            # Subtle.  Same as in the big comment below.
            if id(obj) in self.memo:
                get = self.get(self.memo[id(obj)][0])
                self.write(POP * n + get)
            else:
                self.write(_tuplesize2code[n])
                self.memoize(obj)
            return

        # proto 0 or proto 1 and tuple isn't empty, or proto > 1 and tuple
        # has more than 3 elements.
        self.write(MARK)
        for element in obj:
            self.save(element)

        if id(obj) in self.memo:
            # Subtle.  d was not in memo when we entered save_tuple(), so
            # the process of saving the tuple's elements must have saved
            # the tuple itself:  the tuple is recursive.  The proper action
            # now is to throw away everything we put on the stack, and
            # simply GET the tuple (it's already constructed).  This check
            # could have been done in the "for element" loop instead, but
            # recursive tuples are a rare thing.
            get = self.get(self.memo[id(obj)][0])
            if self.proto:
                self.write(POP_MARK + get)
            else:   # proto 0 -- POP_MARK not available
                self.write(POP * (n+1) + get)
            return

        # No recursion.
        self.write(TUPLE)
        self.memoize(obj)


    # save_empty_tuple() isn't used by anything in Python 2.3.  However, I
    # found a Pickler subclass in Zope3 that calls it, so it's not harmless
    # to remove it.
    def save_empty_tuple(self, obj):
        self.write(EMPTY_TUPLE)
 
    def save_list(self, obj):
        if self.bin:
            self.write(EMPTY_LIST)
        else:   # proto 0 -- can't use EMPTY_LIST
            self.write(MARK + LIST)

        self.memoize(obj)
        self._batch_appends(iter(obj))
    
    def _batch_appends(self, items):
        # Helper to batch up APPENDS sequences
        if not self.bin:
            for x in items:
                self.save(x)
                self.write(APPEND)
            return

        r = xrange(self._BATCHSIZE)
        while items is not None:
            tmp = []
            for i in r:
                try:
                    x = items.next()
                    tmp.append(x)
                except StopIteration:
                    items = None
                    break
            n = len(tmp)
            if n > 1:
                self.write(MARK)
                for x in tmp:
                    self.save(x)
                self.write(APPENDS)
            elif n:
                self.save(tmp[0])
                self.write(APPEND)
            # else tmp is empty, and we're done

    def save_dict(self, obj):
       #print 'saving', obj
        if obj is __builtins__:
            self.save_reduce(_get_module_builtins, (), obj=obj)
            return 

        if self.bin:
            self.write(EMPTY_DICT)
        else:   # proto 0 -- can't use EMPTY_DICT
            self.write(MARK + DICT)

        self.memoize(obj)
        self._batch_setitems(obj.iteritems())

    def _batch_setitems(self, items):
        # Helper to batch up SETITEMS sequences; proto >= 1 only
        if not self.bin:
            for k, v in items:
                self.save(k)
                self.save(v)
                self.write(SETITEM)
            return

        r = xrange(self._BATCHSIZE)
        while items is not None:
            tmp = []
            for i in r:
                try:
                    tmp.append(items.next())
                except StopIteration:
                    items = None
                    break
            n = len(tmp)
            if n > 1:
                self.write(MARK)
                for k, v in tmp:
                    self.save(k)
                    self.save(v)
                self.write(SETITEMS)
            elif n:
                k, v = tmp[0]
                self.save(k)
                self.save(v)
                self.write(SETITEM)
            # else tmp is empty, and we're done
                       
    def save_reduce(self, func, args, state=None,
                    listitems=None, dictitems=None, obj=None):
        """Modified to support __transient__ on new objects
        Change only affects protocol level 2 (which is always used by PiCloud"""
        # Assert that args is a tuple or None
        if not isinstance(args, TupleType):
            raise pickle.PicklingError("args from reduce() should be a tuple")

        # Assert that func is callable
        if not hasattr(func, '__call__'):
            raise pickle.PicklingError("func from reduce should be callable")

        # Protocol 2 special case: if func's name is __newobj__, use NEWOBJ
        if self.proto >= 2 and getattr(func, "__name__", "") == "__newobj__":
            #Added fix to allow transient
            cls = args[0]
            if not hasattr(cls, "__new__"):
                raise pickle.PicklingError(
                    "args[0] from __newobj__ args has no __new__")
            if obj is not None and cls is not obj.__class__:
                raise pickle.PicklingError(
                    "args[0] from __newobj__ args has the wrong class")
            args = args[1:]
            self.save(cls)

            #Don't pickle transient entries
            if hasattr(obj, '__transient__'):
                transient = obj.__transient__
                state = state.copy()

                for k in list(state.keys()):
                    if k in transient:
                        del state[k]

            self.save(args)
            self.write(NEWOBJ)
        else:
            self.save(func)
            self.save(args)
            self.write(REDUCE)

        if obj is not None:
            self.memoize(obj)

        # More new special cases (that work with older protocols as
        # well): when __reduce__ returns a tuple with 4 or 5 items,
        # the 4th and 5th item should be iterators that provide list
        # items and dict items (as (key, value) tuples), or None.

        if listitems is not None:
            self._batch_appends(listitems)

        if dictitems is not None:
            self._batch_setitems(dictitems)

        if state is not None:
            #print 'obj %s has state %s' % (obj, state)
            self.save(state)
            self.write(BUILD)


    def save_xrange(self, obj):
        """Save an xrange object in python 2.5
        Python 2.6 supports this natively
        """
        self.save_reduce(_build_xrange, xrange_params(obj))

    def save_partial(self, obj):
        """Partial objects do not serialize correctly in python2.x -- this fixes the bugs"""
        self.save_reduce(_genpartial, (obj.func, obj.args, obj.keywords))

    def save_file(self, obj):
        """Save a file"""
        import StringIO as pystringIO #we can't use cStringIO as it lacks the name attribute
        from ..transport.adapter import SerializingAdapter

        if not hasattr(obj, 'name') or  not hasattr(obj, 'mode'):
            raise pickle.PicklingError("Cannot pickle files that do not map to an actual file")
        if obj.name == '<stdout>':
            return self.save_reduce(getattr, (sys,'stdout'), obj=obj)
        if obj.name == '<stderr>':
            return self.save_reduce(getattr, (sys,'stderr'), obj=obj)
        if obj.name == '<stdin>':
            raise pickle.PicklingError("Cannot pickle standard input")
        if  hasattr(obj, 'isatty') and obj.isatty():
            raise pickle.PicklingError("Cannot pickle files that map to tty objects")
        if 'r' not in obj.mode:
            raise pickle.PicklingError("Cannot pickle files that are not opened for reading")
        name = obj.name
        try:
            fsize = os.stat(name).st_size
        except OSError:
            raise pickle.PicklingError("Cannot pickle file %s as it cannot be stat" % name)

        if obj.closed:
            #create an empty closed string io
            retval = pystringIO.StringIO("")
            retval.close()
        elif not fsize: #empty file
            retval = pystringIO.StringIO("")
            try:
                tmpfile = file(name)
                tst = tmpfile.read(1)
            except IOError:
                raise pickle.PicklingError("Cannot pickle file %s as it cannot be read" % name)
            tmpfile.close()
            if tst != '':
                raise pickle.PicklingError("Cannot pickle file %s as it does not appear to map to a physical, real file" % name)
        elif fsize > SerializingAdapter.max_transmit_data:
            raise pickle.PicklingError("Cannot pickle file %s as it exceeds cloudconf.py's max_transmit_data of %d" %
                                       (name,SerializingAdapter.max_transmit_data))
        else:
            try:
                tmpfile = file(name)
                contents = tmpfile.read(SerializingAdapter.max_transmit_data)
                tmpfile.close()
            except IOError:
                raise pickle.PicklingError("Cannot pickle file %s as it cannot be read" % name)
            retval = pystringIO.StringIO(contents)
            curloc = obj.tell()
            retval.seek(curloc)

        retval.name = name
        self.save(retval)  #save stringIO
        self.memoize(obj)

    """Special functions for Add-on libraries"""

    def inject_numpy(self):
        numpy = sys.modules.get('numpy')
        if not numpy or not hasattr(numpy, 'ufunc'):
            return
        dispatch[numpy.ufunc] = self.__class__.save_ufunc

    def save_ufunc(self, obj):
        """Hack function for saving numpy ufunc objects"""
        name = obj.__name__
        for tst_mod_name in self.numpy_tst_mods:
            tst_mod = sys.modules.get(tst_mod_name, None)
            if tst_mod:
                if name in tst_mod.__dict__:
                    self.save_reduce(_getobject, (tst_mod_name, name))
                    return
        raise pickle.PicklingError('cannot save %s. Cannot resolve what module it is defined in' % str(obj))

    def inject_timeseries(self):
        """Handle bugs with pickling scikits timeseries"""
        tseries = sys.modules.get('scikits.timeseries.tseries')
        if not tseries or not hasattr(tseries, 'Timeseries'):
            return
        dispatch[tseries.Timeseries] = self.__class__.save_timeseries

    def save_timeseries(self, obj):
        import scikits.timeseries.tseries as ts

        func, reduce_args, state = obj.__reduce__()
        if func != ts._tsreconstruct:
            raise pickle.PicklingError('timeseries using unexpected reconstruction function %s' % str(func))
        state = (1,
                         obj.shape,
                         obj.dtype,
                         obj.flags.fnc,
                         obj._data.tostring(),
                         ts.getmaskarray(obj).tostring(),
                         obj._fill_value,
                         obj._dates.shape,
                         obj._dates.__array__().tostring(),
                         obj._dates.dtype, #added -- preserve type
                         obj.freq,
                         obj._optinfo,
                         )
        return self.save_reduce(_genTimeSeries, (reduce_args, state))

    def inject_email(self):
        """Block email LazyImporters from being saved"""
        email = sys.modules.get('email')
        if not email:
            return
        dispatch[email.LazyImporter] = self.__class__.save_unsupported

    def inject_addons(self):
        """Plug in system. Register additional pickling functions if modules already loaded"""
        self.inject_numpy()
        self.inject_timeseries()
        self.inject_email()

    """Python Imaging Library"""
    def save_image(self, obj):
        if not obj.im and obj.fp and 'r' in obj.fp.mode and obj.fp.name \
            and not obj.fp.closed and (not hasattr(obj, 'isatty') or not obj.isatty()):
            #if image not loaded yet -- lazy load
            self.save_reduce(_lazyloadImage,(obj.fp,), obj=obj)
        else:
            #image is loaded - just transmit it over
            self.save_reduce(_generateImage, (obj.size, obj.mode, obj.tostring()), obj=obj)

    def memoize(self, obj):
        """Store an object in the memo."""

        # The Pickler memo is a dictionary mapping object ids to 2-tuples
        # that contain the Unpickler memo key and the object being memoized.
        # The memo key is written to the pickle and will become
        # the key in the Unpickler's memo.  The object is stored in the
        # Pickler memo so that transient objects are kept alive during
        # pickling.

        # The use of the Unpickler memo length as the memo key is just a
        # convention.  The only requirement is that the memo values be unique.
        # But there appears no advantage to any other scheme, and this
        # scheme allows the Unpickler memo to be implemented as a plain (but
        # growable) array, indexed by memo key.
        if self.fast:
            return
        assert id(obj) not in self.memo
        memo_len = len(self.memo)
        self.write(self.put(memo_len))
        self.memo[id(obj)] = memo_len, obj

# Pickling helpers

cdef void _keep_alive(object x, dict memo):
    """Keeps a reference to the object x in the memo.

    Because we remember objects by their id, we have
    to assure that possibly temporary objects are kept
    alive by referencing them.
    We store a reference at the id of the memo, which should
    normally not be used unless someone tries to deepcopy
    the memo itself...
    """
    try:
        memo[id(memo)].append(x)
    except KeyError:
        # aha, this is the first one :-)
        memo[id(memo)]=[x]


# A cache for whichmodule(), mapping a function object to the name of
# the module in which the function was found.

cdef dict classmap = {} # called classmap for backwards compatibility

def whichmodule(func, funcname):
    """Figure out the module in which a function occurs.

    Search sys.modules for the module.
    Cache in classmap.
    Return a module name.
    If the function cannot be found, return "__main__".
    """
    # Python functions should always get an __module__ from their globals.
    mod = getattr(func, "__module__", None)
    if mod is not None:
        return mod
    if func in classmap:
        return classmap[func]

    for name, module in sys.modules.items():
        if module is None:
            continue # skip dummy package entries
        if name != '__main__' and getattr(module, funcname, None) is func:
            break
    else:
        name = '__main__'
    classmap[func] = name
    return name

# Encode/decode longs in linear time.

import binascii as _binascii

cpdef encode_long(long x):
    """Encode a long to a two's complement little-endian binary string.
    Note that 0L is a special case, returning an empty string, to save a
    byte in the LONG1 pickling context.

    >>> encode_long(0L)
    ''
    >>> encode_long(255L)
    '\xff\x00'
    >>> encode_long(32767L)
    '\xff\x7f'
    >>> encode_long(-256L)
    '\x00\xff'
    >>> encode_long(-32768L)
    '\x00\x80'
    >>> encode_long(-128L)
    '\x80'
    >>> encode_long(127L)
    '\x7f'
    >>>
    """

    if x == 0:
        return ''
    
    cdef str ashex
    cdef int njunkchars, nibbles, nbits, newnibbles
    if x > 0:
        ashex = hex(x)
        assert ashex.startswith("0x")
        njunkchars = 2 + ashex.endswith('L')
        nibbles = len(ashex) - njunkchars
        if nibbles & 1:
            # need an even # of nibbles for unhexlify
            ashex = "0x0" + ashex[2:]
        elif int(ashex[2], 16) >= 8:
            # "looks negative", so need a byte of sign bits
            ashex = "0x00" + ashex[2:]
    else:
        # Build the 256's-complement:  (1L << nbytes) + x.  The trick is
        # to find the number of bytes in linear time (although that should
        # really be a constant-time task).
        ashex = hex(-x)
        assert ashex.startswith("0x")
        njunkchars = 2 + ashex.endswith('L')
        nibbles = len(ashex) - njunkchars
        if nibbles & 1:
            # Extend to a full byte.
            nibbles += 1
        nbits = nibbles * 4
        x += 1L << nbits
        assert x > 0
        ashex = hex(x)
        njunkchars = 2 + ashex.endswith('L')
        newnibbles = len(ashex) - njunkchars
        if newnibbles < nibbles:
            ashex = "0x" + "0" * (nibbles - newnibbles) + ashex[2:]
        if int(ashex[2], 16) < 8:
            # "looks positive", so need a byte of sign bits
            ashex = "0xff" + ashex[2:]

    if ashex.endswith('L'):
        ashex = ashex[2:-1]
    else:
        ashex = ashex[2:]
    assert len(ashex) & 1 == 0, (x, ashex)
    binary = _binascii.unhexlify(ashex)
    return binary[::-1]

# Shorthands for legacy support

cpdef dump(obj, file, int protocol=2):
    CloudPickler(file, protocol).dump(obj)

cpdef dumps(obj, int protocol=2):   
    file = Writer()
    cp = CloudPickler(file,protocol)
    cp.dump(obj)

    #print 'cloud dumped', str(obj), str(cp.modules)

    return file.getvalue()


#hack for __import__ not working as desired
def subimport(name):
    __import__(name)
    return sys.modules[name]

#hack to load django settings:
def django_settings_load(name):
    modified_env = False

    if 'DJANGO_SETTINGS_MODULE' not in os.environ:
        os.environ['DJANGO_SETTINGS_MODULE'] = name # must set name first due to circular deps
        modified_env = True
    try:
        module = subimport(name)
    except Exception, i:
        print >> sys.stderr, 'Cloud not import django settings %s:' % (name)
        print_exec(sys.stderr)
        if modified_env:
            del os.environ['DJANGO_SETTINGS_MODULE']
    else:
        #add project directory to sys,path:
        if hasattr(module,'__file__'):
            dirname = os.path.split(module.__file__)[0] + '/'
            sys.path.append(dirname)

# restores function attributes
def _restore_attr(obj, attr):
    for key, val in attr.items():
        setattr(obj, key, val)
    return obj

def _get_module_builtins():
    return pickle.__builtins__

def print_exec(stream):
    ei = sys.exc_info()
    traceback.print_exception(ei[0], ei[1], ei[2], None, stream)

def _modules_to_main(modList):
    """Force every module in modList to be placed into main"""
    if not modList:
        return

    main = sys.modules['__main__']
    for modname in modList:
        if type(modname) is str:
            try:
                mod = __import__(modname)
            except Exception, i: #catch all...
                sys.stderr.write('warning: could not import %s\n.  Your function may unexpectedly error due to this import failing; \
A version mismatch is likely.  Specific error was:\n' % modname)
                print_exec(sys.stderr)
            else:
                setattr(main,mod.__name__, mod)
        else:
            #REVERSE COMPATIBILITY FOR CLOUD CLIENT 1.5 (WITH EPD)
            #In old version actual module was sent
            setattr(main,modname.__name__, modname)

#object generators:
def _build_xrange(start, step, len):
    """Built xrange explicitly"""
    return xrange(start, start + step*len, step)

def _genpartial(func, args, kwds):
    if not args:
        args = ()
    if not kwds:
        kwds = {}
    return partial(func, *args, **kwds)


def _fill_function(func, globals, defaults, closure, dict):
    """ Fills in the rest of function data into the skeleton function object
        that were created via _make_skel_func().
         """
    func.func_globals.update(globals)
    func.func_defaults = defaults
    func.func_dict = dict

    if len(closure) != len(func.func_closure):
        raise pickle.UnpicklingError("closure lengths don't match up")
    for i in range(len(closure)):
        _change_cell_value(func.func_closure[i], closure[i])

    return func

def _make_skel_func(code, num_closures, base_globals = None):
    """ Creates a skeleton function object that contains just the provided
        code and the correct number of cells in func_closure.  All other
        func attributes (e.g. func_globals) are empty.
    """
    #build closure (cells):
    if not ctypes:
        raise Exception('ctypes failed to import; cannot build function')

    cellnew = ctypes.pythonapi.PyCell_New
    cellnew.restype = ctypes.py_object
    cellnew.argtypes = (ctypes.py_object,)
    dummy_closure = tuple(map(lambda i: cellnew(None), range(num_closures)))

    if base_globals is None:
        base_globals = {}
    base_globals['__builtins__'] = __builtins__

    return FunctionType(code, base_globals,
                              None, None, dummy_closure)

# this piece of opaque code is needed below to modify 'cell' contents
cell_changer_code = new.code(
    1, 1, 2, 0,
    ''.join([
        chr(dis.opmap['LOAD_FAST']), '\x00\x00',
        chr(dis.opmap['DUP_TOP']),
        chr(dis.opmap['STORE_DEREF']), '\x00\x00',
        chr(dis.opmap['RETURN_VALUE'])
    ]),
    (), (), ('newval',), '<nowhere>', 'cell_changer', 1, '', ('c',), ()
)

def _change_cell_value(cell, newval):
    """ Changes the contents of 'cell' object to newval """
    return new.function(cell_changer_code, {}, None, (), (cell,))(newval)

"""Constructors for 3rd party libraries
Note: These can never be renamed due to client compatibility issues"""

def _getobject(modname, attribute):
    mod = __import__(modname)
    return mod.__dict__[attribute]

def _generateImage(size, mode, str_rep):
    """Generate image from string representation"""
    import Image
    i = Image.new(mode, size)
    i.fromstring(str_rep)
    return i

def _lazyloadImage(fp):
    import Image
    fp.seek(0)  #works in almost any case
    return Image.open(fp)

"""Timeseries"""
def _genTimeSeries(reduce_args, state):
    import scikits.timeseries.tseries as ts
    from numpy import ndarray
    from numpy.ma import MaskedArray


    time_series = ts._tsreconstruct(*reduce_args)

    #from setstate modified
    (ver, shp, typ, isf, raw, msk, flv, dsh, dtm, dtyp, frq, infodict) = state
    #print 'regenerating %s' % dtyp

    MaskedArray.__setstate__(time_series, (ver, shp, typ, isf, raw, msk, flv))
    _dates = time_series._dates
    #_dates.__setstate__((ver, dsh, typ, isf, dtm, frq))  #use remote typ
    ndarray.__setstate__(_dates,(dsh,dtyp, isf, dtm))
    _dates.freq = frq
    _dates._cachedinfo.update(dict(full=None, hasdups=None, steps=None,
                                   toobj=None, toord=None, tostr=None))
    # Update the _optinfo dictionary
    time_series._optinfo.update(infodict)
    return time_series
