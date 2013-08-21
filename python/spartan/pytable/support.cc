#include "spartan/pytable/support.h"

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
    Py_IncRef(p);
  }
}

static inline void intrusive_ptr_release(PyObject* p) {
  if (Py_IsInitialized()) {
    GILHelper h;
    Py_DecRef(p);
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
    Log::fatal("Python error, aborting.");
  }
  return result;
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
  RefPtr dumps;

public:
  Pickler() {
    cPickle = check(PyImport_ImportModule("cPickle"));
    cloudpickle = check(
        PyImport_ImportModule("cloud.serialization.cloudpickle"));
    loads = check(PyObject_GetAttrString(cPickle.get(), "loads"));

//    dumps = check(PyObject_GetAttrString(cPickle.get(), "dumps"));
    dumps = check(PyObject_GetAttrString(cloudpickle.get(), "dumps"));
  }

  RefPtr load(const std::string& data) {
    GILHelper lock;
    RefPtr py_str(check(PyString_FromStringAndSize(data.data(), data.size())));
    return RefPtr(
        check(PyObject_CallFunction(loads.get(), W("O"), py_str.get())));
  }

  std::string store(RefPtr p) {
    GILHelper lock;
    RefPtr py_str(
        check(PyObject_CallFunction(dumps.get(), W("Oi"), p.get(), -1)));
    std::string out;
    char* v;
    Py_ssize_t len;
    PyString_AsStringAndSize(py_str.get(), &v, &len);
    out.resize(len);
    memcpy(&out[0], v, len);
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

    RefPtr p = kPickler.load(input);
    *v = p;
    return true;
  }

  static void write_value(Writer *w, const RefPtr& v) {
    w->write_string(kPickler.store(v.get()));
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
    RefPtr result(
        check(PyObject_CallFunction(code.get(), W("Oi"), k.get(), num_shards)));
    return PyInt_AsLong(result.get());
  }
  DECLARE_REGISTRY_HELPER(Sharder, PySharder);
};
DEFINE_REGISTRY_HELPER(Sharder, PySharder);

class PyAccum: public PyInitableT<AccumulatorT<RefPtr> > {
public:
  void accumulate(RefPtr* v, const RefPtr& update) const {
    GILHelper lock;
    RefPtr result(
        check(PyObject_CallFunction(code.get(), W("OO"), v->get(), update.get())));
    *v = result;
  }
  DECLARE_REGISTRY_HELPER(Accumulator, PyAccum);
};
DEFINE_REGISTRY_HELPER(Accumulator, PyAccum);

class PySelector: public PyInitableT<SelectorT<RefPtr, RefPtr> > {
  RefPtr select(const RefPtr& k, const RefPtr& v) {
    GILHelper lock;

    RefPtr result(
        check(PyObject_CallFunction(code.get(), W("OO"), k.get(), v.get())));
    return result;
  }

  DECLARE_REGISTRY_HELPER(Selector, PySelector);
};
DEFINE_REGISTRY_HELPER(Selector, PySelector);

class PyKernel: public Kernel {
public:
  void run() {
    GILHelper lock;
    RefPtr fn(kPickler.load(args()["map_fn"]));
    RefPtr fn_args(kPickler.load(args()["map_args"]));
    PyObject* result(
        check(PyObject_CallFunction(fn.get(), W("lO"), this, fn_args.get())));
    Py_DecRef(result);
  }
};
REGISTER_KERNEL(PyKernel);

void shutdown(Master*h) {
  delete ((Master*) h);
}

void wait_for_workers(Master* m) {
  m->wait_for_workers();
}

Master*get_master(Table* t) {
  return ((PyTable*) t)->master();
}

Table* get_table(Kernel* k, int id) {
  return ((Kernel*) k)->get_table(id);
}

Table* create_table(Master*m, PyObject* sharder, PyObject* accum,
    PyObject* selector) {
  Py_IncRef(sharder);
  Py_IncRef(accum);
  Py_IncRef(selector);

  PySelector* sel = NULL;
  if (selector != Py_None) {
    sel = new PySelector;
  }

  return ((Master*) m)->create_table(
      new PySharder(), new PyAccum(), sel,
      kPickler.store(sharder), kPickler.store(accum), kPickler.store(selector));
}

void destroy_table(Master*, Table*) {
  Log::fatal("Not implemented.");
}

void foreach_shard(Master*m, Table* t, PyObject* fn, PyObject* args) {
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
  PyObject* result = ((PyTable*) t)->get(k).get();
//  Log::info("Result: %s", to_string(result).c_str());
  if (result == NULL) {
    result = Py_None;
  }
  {
    GILHelper lock;
    Py_IncRef(result);
  }
  return result;
}

void update(Table* t, PyObject* k, PyObject* v) {
  {
    GILHelper lock;
    Py_IncRef(k);
    Py_IncRef(v);
  }
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
  {
    GILHelper lock;
    Py_IncRef(((PyTable::Iterator*) i)->key().get());
  }
  return ((PyTable::Iterator*) i)->key().get();
}

PyObject* iter_value(TableIterator* i) {
  {
    GILHelper lock;
    Py_IncRef(((PyTable::Iterator*) i)->value().get());
  }
  return ((PyTable::Iterator*) i)->value().get();
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
  Py_IncRef(p.get());
  return p.get();
}

PyObject* get_accum(Table* t) {
  RefPtr p = ((PyAccum*) t->accum)->code;
  Py_IncRef(p.get());
  return p.get();
}

PyObject* get_selector(Table* t) {
  if (t->selector == NULL) {
    Py_IncRef(Py_None);
    return Py_None;
  }

  RefPtr p = ((PySelector*) t->selector)->code;
  Py_IncRef(p.get());
  return p.get();
}

}
