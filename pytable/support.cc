#include "pytable/support.h"

#include "sparrow/table.h"
#include "sparrow/master.h"
#include "sparrow/kernel.h"
#include "sparrow/worker.h"

#include "sparrow/util/marshal.h"

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

static inline void intrusive_ptr_add_ref(PyObject* p) {
  Py_IncRef(p);
}

static inline void intrusive_ptr_release(PyObject* p) {
  Py_DecRef(p);
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
  RefPtr loads;
  RefPtr dumps;

public:
  Pickler() {
    cPickle = check(PyImport_ImportModule("cPickle"));
    loads = check(PyObject_GetAttrString(cPickle.get(), "loads"));
    dumps = check(PyObject_GetAttrString(cPickle.get(), "dumps"));
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

namespace sparrow {

typedef TableT<RefPtr, RefPtr> PyTable;

class PythonKernel;
PythonKernel* active_kernel;

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
class PyInitable: public T {
protected:
  RefPtr code_;

public:
  void init(const std::string& opts) {
    code_ = kPickler.load(opts);
  }
};

class PySharder: public PyInitable<SharderT<RefPtr> > {
public:
  size_t shard_for_key(const RefPtr& k, int num_shards) const {
    GILHelper lock;
    RefPtr result(
        PyObject_CallFunction(code_.get(), W("Oi"), k.get(), num_shards));
    return PyInt_AsLong(result.get());
  }
  DECLARE_REGISTRY_HELPER(Sharder, PySharder);
};
DEFINE_REGISTRY_HELPER(Sharder, PySharder);

class PyAccum: public PyInitable<AccumulatorT<RefPtr> > {
private:
  RefPtr code_;
public:
  void init(const std::string& opts) {
    code_ = kPickler.load(opts);
  }

  void accumulate(RefPtr* v, const RefPtr& update) const {
    GILHelper lock;
    RefPtr result(
        PyObject_CallFunction(code_.get(), W("OO"), v->get(), update.get()));
    *v = result;
  }
  DECLARE_REGISTRY_HELPER(Accumulator, PyAccum);
};
DEFINE_REGISTRY_HELPER(Accumulator, PyAccum);

class PySelector: public PyInitable<SelectorT<RefPtr, RefPtr> > {
private:
  RefPtr code_;
public:
  void init(const std::string& opts) {
    code_ = kPickler.load(opts);
  }

  RefPtr select(const RefPtr& k, const RefPtr& v) {
    GILHelper lock;

    RefPtr result(
        check(PyObject_CallFunction(code_.get(), W("OO"), k.get(), v.get())));
    return result;
  }

  DECLARE_REGISTRY_HELPER(Selector, PySelector);
};
DEFINE_REGISTRY_HELPER(Selector, PySelector);

class PythonKernel: public Kernel {
public:
  void run() {
    GILHelper lock;
    RefPtr fn(kPickler.load(args()["map_fn"]));
    RefPtr fn_args(kPickler.load(args()["map_args"]));
    PyObject* result(check(PyObject_CallFunction(fn.get(), W("lO"), this, fn_args.get())));
    Py_DecRef(result);
  }
};
REGISTER_KERNEL(PythonKernel);

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

  return ((Master*) m)->create_table(new PySharder(), new PyAccum(), sel,
      kPickler.store(sharder), kPickler.store(accum), kPickler.store(selector));
}

void destroy_table(Master*, Table*) {
  Log::fatal("Not implemented.");
}

void foreach_shard(Master*m, Table* t, PyObject* fn, PyObject* args) {
  sparrow::RunDescriptor r;
  r.kernel = "PythonKernel";
  r.args["map_fn"] = kPickler.store(fn);
  r.args["map_args"] = kPickler.store(args);
  r.table = (PyTable*) t;
  r.shards = sparrow::range(0, ((Table*) t)->num_shards());
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
  Py_IncRef(result);
  return result;
}

void update(Table* t, PyObject* k, PyObject* v) {
  Py_IncRef(k);
  Py_IncRef(v);
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
  Py_IncRef(((PyTable::Iterator*) i)->key().get());
  return ((PyTable::Iterator*) i)->key().get();
}

PyObject* iter_value(TableIterator* i) {
  Py_IncRef(((PyTable::Iterator*) i)->value().get());
  return ((PyTable::Iterator*) i)->value().get();
}

bool iter_done(TableIterator* i) {
  return ((PyTable::Iterator*) i)->done();
}

void iter_next(TableIterator* i) {
  ((PyTable::Iterator*) i)->next();
}

int current_table(Kernel* k) {
  return ((PythonKernel*) k)->table_id();
}

int current_shard(Kernel* k) {
  return ((PythonKernel*) k)->shard_id();
}

}
