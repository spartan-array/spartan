#include "spartan/support.h"

#include "spartan/table.h"
#include "spartan/master.h"
#include "spartan/kernel.h"
#include "spartan/worker.h"

#include "spartan/util/common.h"
#include "spartan/util/marshal.h"

#include <Python.h>

using rpc::Log;

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
    GILHelper h;
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

static std::string to_string(RefPtr p) {
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
    cPickle = to_ref(PyImport_ImportModule("cPickle"));
    cloudpickle = to_ref(
        PyImport_ImportModule("cloud.serialization.cloudpickle"));
    loads = to_ref(PyObject_GetAttrString(cPickle.get(), "loads"));
    cpickle_dumps = to_ref(PyObject_GetAttrString(cPickle.get(), "dumps"));
    cloud_dumps = to_ref(PyObject_GetAttrString(cloudpickle.get(), "dumps"));
  }

  RefPtr load(const std::string& data) {
    GILHelper lock;
    RefPtr py_str = to_ref(PyString_FromStringAndSize(data.data(), data.size()));
    return to_ref(PyObject_CallFunction(loads.get(), W("O"), py_str.get()));
  }

  std::string store(const RefPtr& p) {
    GILHelper lock;
    PyObject* py_str =
        PyObject_CallFunction(cpickle_dumps.get(), W("Oi"), p.get(), -1);

    if (py_str == NULL) {
      PyErr_Clear();
      py_str = check(PyObject_CallFunction(cloud_dumps.get(), W("Oi"), p.get(), -1));
    }

    std::string out;
    char* v;
    Py_ssize_t len;
    PyString_AsStringAndSize(py_str, &v, &len);
    out.resize(len);
    memcpy(&out[0], v, len);

    Py_XDECREF(py_str);
    return out;
  }
};

static Pickler kPickler;

namespace spartan {

typedef TableT<RefPtr, RefPtr> PyTable;

class PyKernel;
PyKernel* active_kernel;

template<>
class Marshal<RefPtr> {
public:
  static bool read_value(Reader *r, RefPtr* v) {
    std::string input;
    if (!r->read_string(input)) {
      return false;
    }

    *v = kPickler.load(input);
    return true;
  }

  static void write_value(Writer *w, const RefPtr& v) {
    std::string str = kPickler.store(v.get());
    w->write_string(str);
  }
};

template<class T>
class PyInitableT: public T {
public:
  RefPtr code;

  void init(const std::string& opts) {
    code = kPickler.load(opts);
  }
};

class PySharder: public PyInitableT<SharderT<RefPtr> > {
public:
  size_t shard_for_key(const RefPtr& k, int num_shards) const {
    GILHelper lock;
    RefPtr result =
        to_ref(PyObject_CallFunction(code.get(), W("Oi"), k.get(), num_shards));

    CHECK(PyInt_Check(result.get()));
    return PyInt_AsLong(result.get());
  }
  DECLARE_REGISTRY_HELPER(Sharder, PySharder);
};
DEFINE_REGISTRY_HELPER(Sharder, PySharder);

class PyAccum: public PyInitableT<AccumulatorT<RefPtr> > {
public:
  void accumulate(RefPtr* v, const RefPtr& update) const {
    GILHelper lock;
    RefPtr result =
        to_ref(PyObject_CallFunction(code.get(), W("OO"), v->get(), update.get()));
    CHECK(result.get() != Py_None);

    *v = result;
  }
  DECLARE_REGISTRY_HELPER(Accumulator, PyAccum);
};
DEFINE_REGISTRY_HELPER(Accumulator, PyAccum);

class PySelector: public PyInitableT<SelectorT<RefPtr, RefPtr> > {
  RefPtr select(const RefPtr& k, const RefPtr& v) {
    GILHelper lock;

    RefPtr result =
        to_ref(PyObject_CallFunction(code.get(), W("OO"), k.get(), v.get()));
    return result;
  }

  DECLARE_REGISTRY_HELPER(Selector, PySelector);
};
DEFINE_REGISTRY_HELPER(Selector, PySelector);

class PyKernel: public Kernel {
public:
  void run() {
    GILHelper lock;
    RefPtr fn = kPickler.load(args()["map_fn"]);
    RefPtr fn_args = kPickler.load(args()["map_args"]);
    to_ref(PyObject_CallFunction(fn.get(), W("lO"), this, fn_args.get()));
  }
};
REGISTER_KERNEL(PyKernel);

void shutdown(Master*h) {
  delete ((Master*) h);
}

void wait_for_workers(Master* m) {
  m->wait_for_workers();
}

Table* get_table(Kernel* k, int id) {
  return ((Kernel*) k)->get_table(id);
}

Table* create_table(Master*m, PyObject* sharder, PyObject* accum,
    PyObject* selector) {
  Py_XINCREF(sharder);
  Py_XINCREF(accum);
  Py_XINCREF(selector);

  PySelector* sel = NULL;
  if (selector != Py_None) {
    sel = new PySelector;
  }

  return ((Master*) m)->create_table(
      new PySharder(), new PyAccum(), sel,
      kPickler.store(sharder), kPickler.store(accum), kPickler.store(selector));
}

void destroy_table(Master* m, Table* t) {
  m->destroy_table(t);
}

void foreach_shard(Master*m, Table* t,
                   PyObject* fn, PyObject* args) {
  spartan::RunDescriptor r;
  r.kernel = "PyKernel";
  r.args["map_fn"] = kPickler.store(fn);
  r.args["map_args"] = kPickler.store(args);
  r.table = (PyTable*) t;
  r.shards = spartan::range(0, ((Table*) t)->num_shards());
  ((Master*) m)->run(r);
}

int get_id(Table* t) {
  return ((PyTable*) t)->id();
}

PyObject* get(Table* t, PyObject* k) {
  RefPtr result = ((PyTable*) t)->get(k);
//  Log_info("Result: %s", to_string(result).c_str());
  if (result.get() == NULL) {
    result = Py_None;
  }

  Py_XINCREF(result.get());
  return result.get();
}

void update(Table* t, PyObject* k, PyObject* v) {
  ((PyTable*) t)->update(k, v);
}

int num_shards(Table* t) {
  return t->num_shards();
}

TableIterator* get_iterator(Table* t, int shard) {
  if (shard != -1) {
    return ((PyTable*) t)->get_iterator(shard);
  }
  return ((PyTable*) t)->get_iterator();
}

PyObject* iter_key(TableIterator* i) {
  RefPtr k = ((PyTable::Iterator*) i)->key();
  Py_XINCREF(k.get());
  return k.get();
}

PyObject* iter_value(TableIterator* i) {
  RefPtr v = ((PyTable::Iterator*) i)->value();
  Py_XINCREF(v.get());
  return v.get();
}

bool iter_done(TableIterator* i) {
  return ((PyTable::Iterator*) i)->done();
}

void iter_next(TableIterator* i) {
  ((PyTable::Iterator*) i)->next();
}

int current_table(Kernel* k) {
  return ((PyKernel*) k)->table_id();
}

int current_shard(Kernel* k) {
  return ((PyKernel*) k)->shard_id();
}

TableContext* get_context() {
  return TableContext::get_context();
}

Table* get_table(TableContext* t, int id) {
  return t->get_table(id);
}

int get_table_id(Table* t) {
  return t->id();
}

PyObject* get_sharder(Table* t) {
  RefPtr p = ((PySharder*) t->sharder)->code;
  Py_XINCREF(p.get());
  return p.get();
}

PyObject* get_accum(Table* t) {
  RefPtr p = ((PyAccum*) t->accum)->code;
  Py_XINCREF(p.get());
  return p.get();
}

PyObject* get_selector(Table* t) {
  if (t->selector == NULL) {
    Py_INCREF(Py_None);
    return Py_None;
  }

  RefPtr p = ((PySelector*) t->selector)->code;
  Py_XINCREF(p.get());
  return p.get();
}

void set_log_level(LogLevel l) {
  rpc::Log::set_level(l);
}

void log(const char* file, int line, const char* msg) {
  rpc::Log::log(rpc::Log::INFO, file, line, msg);
}

Master* cast_to_master(TableContext* ctx) {
  Master* m = dynamic_cast<Master*>(ctx);
  if (m == NULL) {
    Log_fatal("Tried to cast %p to Master, but not of Master type.");
  }

  return m;
}

}
