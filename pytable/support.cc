#include "pytable/support.h"

#include "sparrow/table.h"
#include "sparrow/master.h"
#include "sparrow/kernel.h"
#include "sparrow/worker.h"

#include "sparrow/util/marshal.h"

#include <Python.h>

namespace sparrow {
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

void foreach_shard(Master* m, Table* t, PyObject* fn, PyObject* args) {
  sparrow::RunDescriptor r;
  r.kernel = "PythonKernel";
  r.args["map_fn"] = pickle(fn);
  r.args["map_args"] = pickle(args);
  r.table = t;
  r.shards = sparrow::range(0, t->num_shards());
  m->run(r);
}

Master* init(int argc, char* argv[]) {
  Init(argc, argv);
  if (!StartWorker()) {
    return new Master();
  }
  return NULL;
}

TableT<RefPtr, RefPtr>* create_table(Master* m, PyObject* sharder,
    PyObject* accum) {
  Py_IncRef(sharder);
  Py_IncRef(accum);
  return m->create_table(new PySharder(), new PyAccum(), pickle(sharder),
      pickle(accum));
}

PyTable* get_table(Kernel* k, int id) {
  return (PyTable*) ((k->get_table(id)));
}

// This is a round-about way of getting SWIG to wrap some function
// arguments for us.
Kernel* as_kernel(long ptr) {
  return ((Kernel*) (ptr));
}

}
