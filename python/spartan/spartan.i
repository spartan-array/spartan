%module spartan_wrap

%include <std_map.i>
%include <std_vector.i>
%include <std_string.i>

%include "numpy.i"

%{
#include "spartan/table.h"
#include "spartan/master.h"
#include "spartan/kernel.h"
#include "spartan/worker.h"

#include "spartan/util/common.h"
#include "spartan/util/marshal.h"

#include "Python.h"
#include <string>
  
  
 // Python utility functions/classes
struct GILHelper {
  PyGILState_STATE gstate;
  GILHelper() {
    gstate = PyGILState_Ensure();
  }

  ~GILHelper() {
    PyGILState_Release(gstate);
  }
};

// Manage Python reference counts.  If Python is shutting
// down, then the GIL is no longer valid so we don't do
// anything (and we don't care about ref counts at that point anyway).
static inline void intrusive_ptr_add_ref(PyObject* p) {
  if (Py_IsInitialized()) {
  //  GILHelper h;
    Py_XINCREF(p);
  }
}

static inline void intrusive_ptr_release(PyObject* p) {
  if (Py_IsInitialized()) {
    GILHelper h;
    Py_XDECREF(p);
  }
}

typedef boost::intrusive_ptr<PyObject> RefPtr;

static inline std::string to_string(RefPtr p) {
  RefPtr py_str(PyObject_Str(p.get()));

  char* c_str;
  Py_ssize_t c_len;

  PyString_AsStringAndSize(py_str.get(), &c_str, &c_len);

  return std::string(c_str, c_len);
}

// Get rid of annoying writable string warnings.
#define W(str) (char*)str

template<class T>
T check(T result) {
  if (PyErr_Occurred()) {
    PyErr_Print();
    spartan::print_backtrace();
    Log_fatal("Python error, aborting.");
  }
  return result;
}

std::string repr(RefPtr p) {
  GILHelper h;
  PyObject* str = PyObject_Repr(p.get());
  char* data;
  Py_ssize_t sz;
  PyString_AsStringAndSize(str, &data, &sz);
  Py_DECREF(str);
  
  std::string out(data, sz);
  
  return out;
}

// Most python functions return 'new' references; we don't need
// to incref these when turning them into a RefPtr.
RefPtr to_ref(PyObject* o) {
  return RefPtr(check(o), false);
}

namespace boost {
static inline size_t hash_value(const RefPtr& p) {
  GILHelper lock;
  return check(PyObject_Hash(p.get()));
}
} // namespace boost

static inline bool operator==(const RefPtr& a, const RefPtr& b) {
  GILHelper lock;
  return check(PyObject_Compare(a.get(), b.get())) == 0;
}

class Pickler {
  RefPtr cPickle;
  RefPtr cloudpickle;
  RefPtr loads;
  RefPtr cloud_dumps;
  RefPtr cpickle_dumps;

public:
  Pickler() {
    GILHelper lock;
    cPickle = to_ref(PyImport_ImportModule("cPickle"));
    cloudpickle = to_ref(PyImport_ImportModule("spartan.cloudpickle"));
    loads = to_ref(PyObject_GetAttrString(cPickle.get(), "loads"));
    cpickle_dumps = to_ref(PyObject_GetAttrString(cPickle.get(), "dumps"));
    cloud_dumps = to_ref(PyObject_GetAttrString(cloudpickle.get(), "dumps"));
  }

  RefPtr load(const RefPtr& py_str) {
    GILHelper lock;
    //Log_info("Loading %d bytes", PyString_Size(py_str.get()));
    return to_ref(PyObject_CallFunction(loads.get(), W("O"), py_str.get()));
  }

  RefPtr load(const std::string& data) {
    GILHelper lock;
    RefPtr py_str = to_ref(
        PyString_FromStringAndSize(data.data(), data.size()));
    return load(py_str);
  }

  void store(spartan::Writer* w, const RefPtr& p) {
    GILHelper lock;

    // cPickle is faster, but will fail when trying to serialize lambdas/closures
    // try it first and fall back to cloudpickle.
    PyObject* py_str = PyObject_CallFunction(
        cpickle_dumps.get(), W("Oi"), p.get(), -1);

    if (py_str == NULL) {
      PyErr_Clear();
      py_str = check(
          PyObject_CallFunction(cloud_dumps.get(), W("Oi"), p.get(), -1));
    }

    char* v;
    Py_ssize_t len;
    PyString_AsStringAndSize(py_str, &v, &len);
    w->write_bytes(v, len);

    Py_DECREF(py_str);

    if (len > 1e6) {
      Log_info("Large value for pickle: %d", len);
      //abort();
    }
  }

  std::string store(const RefPtr& p) {
    std::string out;
    spartan::StringWriter w(&out);
    store(&w, p);
    return out;
  }
};

static Pickler* _pickler = NULL;
static Pickler& get_pickler() {
  if (_pickler == NULL) {
    _pickler = new Pickler;
  }
  return *_pickler;
}

template<class T>
static inline PyObject* get_code(T* obj) {
  if (obj == NULL) {
    Py_INCREF(Py_None);
    return Py_None;
  }

  RefPtr p = obj->code;
  Py_INCREF(p.get());
  return p.get();
}

namespace spartan {
  
template<>
class Marshal<RefPtr> {
public:
  static bool read_value(Reader *r, RefPtr* v) {
    RefPtr py_str;
    {
      GILHelper lock;
      py_str = to_ref(PyString_FromStringAndSize(NULL, r->bytes_left()));
    }

    if (!r->read_bytes(PyString_AsString(py_str.get()), r->bytes_left())) {
      return false;
    }

    *v = get_pickler().load(py_str);
    return true;
  }

  static void write_value(Writer *w, const RefPtr& v) {
    get_pickler().store(w, v);
  }
};

template<class T>
class PyInitableT: public T {
public:
  RefPtr code;
  std::string opts_;

  void init(const std::string& opts) {
    opts_ = opts;
    code = get_pickler().load(opts);
  }

  PyInitableT() {
  }
  PyInitableT(const std::string& opts) {
    init(opts);
  }

  const std::string& opts() {
    return opts_;
  }
};

class PySharder: public PyInitableT<SharderT<RefPtr> > {
public:
  size_t shard_for_key(const RefPtr& k, int num_shards) const {
    GILHelper lock;
    RefPtr result = to_ref(
        PyObject_CallFunction(code.get(), W("Oi"), k.get(), num_shards));

    CHECK(PyInt_Check(result.get()));
    return PyInt_AsLong(result.get());
  }
  DECLARE_REGISTRY_HELPER(Sharder, PySharder);
};
DEFINE_REGISTRY_HELPER(Sharder, PySharder);

class PyAccum: public PyInitableT<AccumulatorT<RefPtr, RefPtr> > {
public:
  void accumulate(const RefPtr& k, RefPtr* v, const RefPtr& update) const {
    GILHelper lock;
    RefPtr result = to_ref(
        PyObject_CallFunction(code.get(), W("OOO"), k.get(), v->get(),
            update.get()));
    CHECK(result.get() != Py_None);
    CHECK(result.get() != NULL);

    *v = result;
  }
  DECLARE_REGISTRY_HELPER(Accumulator, PyAccum);
};
DEFINE_REGISTRY_HELPER(Accumulator, PyAccum);

class PySelector: public PyInitableT<SelectorT<RefPtr, RefPtr> > {
  RefPtr select(const RefPtr& k, const RefPtr& v) {
    GILHelper lock;

    RefPtr result = to_ref(
        PyObject_CallFunction(code.get(), W("OO"), k.get(), v.get()));
    return result;
  }

  DECLARE_REGISTRY_HELPER(Selector, PySelector);
};
DEFINE_REGISTRY_HELPER(Selector, PySelector);

class PyKernel: public Kernel {
public:
  void run() {
    GILHelper lock;
    RefPtr fn = get_pickler().load(args()["map_fn"]);
    to_ref(PyObject_CallFunction(fn.get(), W("l"), this));
  }
  DECLARE_REGISTRY_HELPER(Kernel, PyKernel);
};
DEFINE_REGISTRY_HELPER(Kernel, PyKernel);

typedef TableT<RefPtr, RefPtr> PyTable;
typedef TypedIterator<RefPtr, RefPtr> PyIterator;

} // namespace spartan

using namespace spartan;

%} // end helpers


typedef boost::intrusive_ptr<PyObject> RefPtr;
namespace spartan {

%typemap(in) RefPtr {
  $1 = RefPtr($input);
}

%typemap(out) RefPtr {
  if ($1.get() == NULL) {
    $result = Py_None;
  } else {
    $result = $1.get();
  }
  
  Py_XINCREF($result);
  return $result;
}

class PyIterator {
private:
  PyIterator();
public:
  RefPtr key();
  RefPtr value();
  int shard();
  bool done();
  void next();
};

class PyTable {
public:
  int id();
  int num_shards();

  RefPtr get(int shard, RefPtr k);
  void update(int shard, RefPtr k, RefPtr v);

  PyIterator* get_iterator();
  PyIterator* get_iterator(int shard);
};

%extend PyTable {
  PyTable(int id) {
    return (PyTable*)TableContext::get_context()->get_table(id);
  }
  
  PyObject* combiner() {
    return get_code((PyAccum*) $self->combiner);
  }
  
  PyObject* reducer() {
    return get_code((PyAccum*) $self->reducer);
  }
  
  PyObject* sharder() {
    return get_code((PySharder*) $self->sharder);
  }
  
  PyObject* selector() {
    return get_code((PySelector*) $self->selector);
  }
}

class TableContext {
private:
  TableContext();
public:
  static TableContext* get_context();
};

class Kernel {
private:
  Kernel();
public:
  
};

%extend Kernel {
  PyObject* args() {
    GILHelper h;
    PyObject* out = PyDict_New();
    for (auto i : $self->args()) {
      PyDict_SetItemString(out, i.first.c_str(),
          PyString_FromStringAndSize(i.second.data(), i.second.size()));
    }
  
    return out;
  }
  
  PyTable* table(int id) {
    return (PyTable*)$self->get_table(id);
  }
}

class Worker {
  Worker();
public:
  void wait_for_shutdown();
};

class Master {
private:
  Master();
public:
  void shutdown();
  void wait_for_workers();
  int num_workers();
  void destroy_table(PyTable*);
};

%extend Master {
  PyTable* create_table(PyObject* sharder, PyObject* combiner, PyObject* reducer, PyObject* selector) {
    Py_XINCREF(sharder);
    Py_XINCREF(combiner);
    Py_XINCREF(reducer);
    Py_XINCREF(selector);
  
    Pickler& p = get_pickler();
  
    auto py_sharder = Initable::create<PySharder>(p.store(sharder));
  
    PyAccum* py_combiner = NULL;
    if (combiner != Py_None) {
      py_combiner = Initable::create<PyAccum>(p.store(combiner));
    }
  
    PyAccum* py_reducer = NULL;
    if (reducer != Py_None) {
      py_reducer = Initable::create<PyAccum>(p.store(reducer));
    }
  
    PySelector* py_selector = NULL;
    if (selector != Py_None) {
      py_selector = Initable::create<PySelector>(p.store(selector));
    }
  
    return $self->create_table(py_sharder, py_combiner, py_reducer, py_selector);
  }
  
  void foreach_shard(PyTable* t, PyObject* fn, PyObject* args) {
    auto p = new PyKernel();
    p->args()["map_fn"] = get_pickler().store(fn);
    p->args()["map_args"] = get_pickler().store(args);
    $self->map_shards(t, p);
  }

  void foreach_worklist(PyObject* worklist, PyObject* fn) {
    auto p = new PyKernel();
    p->args()["map_fn"] = get_pickler().store(fn);
    
    WorkList w;
    {
      GILHelper h;
      auto iter = to_ref(check(PyObject_GetIter(worklist)));
      while (1) {
        auto py_item = to_ref(PyIter_Next(iter.get()));
        if (py_item.get() == NULL) {
          break;
        }
        
        PyObject* args = check(PyTuple_GetItem(py_item.get(), 0));
  
        int table, shard;
        PyObject* locality = check(PyTuple_GetItem(py_item.get(), 1));
        check(PyArg_ParseTuple(locality, "ii", &table, &shard));
        
        WorkItem w_item;
        w_item.args["map_args"] = get_pickler().store(args);
        w_item.locality = ShardId(table, shard);
        w.push_back(w_item);
      }
    }
    
    $self->map_worklist(w, p);
  }
} // extend Master


Master* start_master(int port, int num_workers);
Worker* start_worker(const std::string& master, int port);

} // namespace spartan(handle


%inline %{
  
namespace spartan {

enum LogLevel {
    FATAL = 0, ERROR = 1, WARN = 2, INFO = 3, DEBUG = 4
};

void set_log_level(int l) {
  rpc::Log::set_level(l);
}

void log(int level, const char* file, int line, const char* msg) {
  rpc::Log::log(level, line, file, msg);
}

Master* cast_to_master(TableContext* ctx) {
  Master* m = dynamic_cast<Master*>(ctx);
  if (m == NULL) {
    Log_fatal("Tried to cast %p to Master, but not of Master type.");
  }

  return m;
}

// Hack to allow passing Kernel* to user functions.
static inline Kernel* cast_to_kernel(long kernel_handle) {
  return (Kernel*)kernel_handle;
}

static int x;
PyTable* blah() {
  return (PyTable*)(&x);
}

} // namespace spartan

%} // inline
