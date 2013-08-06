#include "pytable/support.h"

#include "sparrow/table.h"
#include "sparrow/master.h"
#include "sparrow/kernel.h"
#include "sparrow/worker.h"

#include "sparrow/util/marshal.h"

#include <Python.h>

static inline void intrusive_ptr_add_ref(PyObject* p) {
  Py_IncRef(p);
}

static inline void intrusive_ptr_release(PyObject* p) {
  Py_DecRef(p);
}

typedef boost::intrusive_ptr<PyObject> RefPtr;

namespace boost {
static inline size_t hash_value(const RefPtr& p) {
  return PyObject_Hash(p.get());
}
}

static inline bool operator==(const RefPtr& a, const RefPtr& b) {
  return PyObject_Compare(a.get(), b.get()) == 0;
}

namespace sparrow {

typedef TableT<RefPtr, RefPtr> PyTable;

class PythonKernel;

PythonKernel* active_kernel;

PyObject* check(PyObject* result) {
  if (PyErr_Occurred()) {
    PyErr_Print();
    LOG(FATAL)<< "Python error, aborting.";
    return NULL;
  }
  return result;
}

RefPtr unpickle(const std::string& data) {
  Timer t;
  RefPtr globals(check(PyEval_GetGlobals()));
  RefPtr cPickle(check(PyImport_AddModule("cPickle")));
  RefPtr loads(check(PyObject_GetAttrString(cPickle.get(), "loads")));
  RefPtr py_str(check(PyString_FromStringAndSize(data.data(), data.size())));
//  LOG(INFO) << "Unpickle took " << t.elapsed() << " seconds.";
  return RefPtr(check(PyObject_CallFunction(loads.get(), "O", py_str.get())));
}

std::string pickle(RefPtr p) {
  Timer t;
  RefPtr globals(check(PyEval_GetGlobals()));
  RefPtr cPickle(check(PyImport_AddModule("cPickle")));
  RefPtr dumps(check(PyObject_GetAttrString(cPickle.get(), "dumps")));
  RefPtr py_str(check(PyObject_CallFunction(dumps.get(), "Oi", p.get(), -1)));
  std::string out;
  char* v;
  Py_ssize_t len;
  PyString_AsStringAndSize(py_str.get(), &v, &len);
  out.resize(len);
  memcpy(&out[0], v, len);
//  LOG(INFO) << "Pickle took " << t.elapsed() << " seconds.";
  return out;
}

template<>
class Marshal<RefPtr> {
public:
  static bool read_value(Reader *r, RefPtr* v) {
    std::string input;
    if (!r->read_string(input)) {
      return false;
    }

    RefPtr p = unpickle(input);
    *v = p;
    return true;
  }

  static void write_value(Writer *w, const RefPtr& v) {
    w->write_string(pickle(v.get()));
  }
};

class PySharder: public SharderT<RefPtr> {
private:
  RefPtr code_;

public:
  void init(const std::string& opts) {
    code_ = unpickle(opts);
  }

  size_t shard_for_key(const RefPtr& k, int num_shards) const {
    RefPtr result(
        PyObject_CallFunction(code_.get(), "Oi", k.get(), num_shards));
    return PyInt_AsLong(result.get());
  }
  DECLARE_REGISTRY_HELPER(Sharder, PySharder);
};
DEFINE_REGISTRY_HELPER(Sharder, PySharder);

class PyAccum: public AccumulatorT<RefPtr> {
private:
  RefPtr code_;
public:
  void init(const std::string& opts) {
    code_ = unpickle(opts);
  }

  void accumulate(RefPtr* v, const RefPtr& update) const {
    RefPtr result(
        PyObject_CallFunction(code_.get(), "OO", v->get(), update.get()));
    *v = result;
  }
  DECLARE_REGISTRY_HELPER(Accumulator, PyAccum);
};
DEFINE_REGISTRY_HELPER(Accumulator, PyAccum);

class PythonKernel: public Kernel {
public:
  void run() {
    RefPtr fn(unpickle(args()["map_fn"]));
    RefPtr fn_args(unpickle(args()["map_args"]));
    PyObject_CallFunction(fn.get(), "lO", this, fn_args.get());
  }
};
REGISTER_KERNEL(PythonKernel);

MasterHandle init(int argc, char* argv[]) {
  Init(argc, argv);
  if (!StartWorker()) {
    return (MasterHandle)(new Master());
  }
  return NULL;
}

void shutdown(MasterHandle h) {
  delete ((Master*)h);
}

MasterHandle get_master(TableHandle t) {
  return (MasterHandle)((PyTable*)t)->master();
}

TableHandle get_table(KernelHandle k, int id) {
  return (TableHandle)((Kernel*) k)->get_table(id);
}

TableHandle create_table(MasterHandle m, PyObject* sharder, PyObject* accum) {
  Py_IncRef(sharder);
  Py_IncRef(accum);
  return (TableHandle)((Master*) m)->create_table(new PySharder(), new PyAccum(),
      pickle(sharder), pickle(accum));
}

void destroy_table(MasterHandle, TableHandle) {
  LOG(FATAL)<< "Not implemented.";
}

void foreach_shard(MasterHandle m, TableHandle t, PyObject* fn,
    PyObject* args) {
  sparrow::RunDescriptor r;
  r.kernel = "PythonKernel";
  r.args["map_fn"] = pickle(fn);
  r.args["map_args"] = pickle(args);
  r.table = (PyTable*)t;
  r.shards = sparrow::range(0, ((Table*)t)->num_shards());
  ((Master*)m)->run(r);
}

int get_id(TableHandle t) {
  return ((PyTable*)t)->id();
}

PyObject* get(TableHandle t, PyObject* k) {
  PyObject* result = ((PyTable*)t)->get(k).get();
  if (result == NULL) {
    result = Py_None;
  }
  Py_IncRef(result);
  return result;
}

void update(TableHandle t, PyObject* k, PyObject* v) {
  Py_IncRef(k);
  Py_IncRef(v);
  ((PyTable*)t)->update(k, v);
}

IteratorHandle get_iterator(TableHandle t, int shard) {
  if (shard != -1) {
    return (IteratorHandle)((PyTable*)t)->get_iterator(shard);
  }
  return (IteratorHandle)((PyTable*)t)->get_iterator();
}

PyObject* iter_key(IteratorHandle i) {
  Py_IncRef(((PyTable::Iterator*)i)->key().get());
  return ((PyTable::Iterator*)i)->key().get();
}

PyObject* iter_value(IteratorHandle i) {
  Py_IncRef(((PyTable::Iterator*)i)->value().get());
  return ((PyTable::Iterator*)i)->value().get();
}

bool iter_done(IteratorHandle i) {
  return ((PyTable::Iterator*)i)->done();
}

void iter_next(IteratorHandle i) {
  ((PyTable::Iterator*)i)->next();
}

int current_table(KernelHandle k) {
  return ((PythonKernel*)k)->table_id();
}

int current_shard(KernelHandle k) {
  return ((PythonKernel*)k)->shard_id();
}

}
