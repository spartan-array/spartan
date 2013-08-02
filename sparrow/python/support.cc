#include "sparrow/python/support.h"

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
  LOG(INFO) << "Unpickle took " << t.elapsed() << " seconds.";
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
  LOG(INFO) << "Pickle took " << t.elapsed() << " seconds.";
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
    PyObject_CallFunction(code_.get(), "OO", v->get(), update.get());
  }

  DECLARE_REGISTRY_HELPER(Accumulator, PyAccum);
};
DEFINE_REGISTRY_HELPER(Accumulator, PyAccum);

class PythonKernel: public Kernel {
public:

  std::string code() {
    return args()["map_fn"];
  }

  void run() {
    active_kernel = this;
    PyRun_SimpleString("import sparrow; sparrow._bootstrap_kernel()");
  }
};
REGISTER_KERNEL(PythonKernel);

void _map_shards(Master* m, Table* t, const std::string& fn,
    const std::string& args) {
  sparrow::RunDescriptor r;
  r.kernel = "PythonKernel";
  r.args["map_fn"] = fn;
  r.args["map_args"] = args;
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

PyTable* get_table(int id) {
  return (PyTable*)active_kernel->get_table(id);
}

int current_table_id() {
  return active_kernel->table_id();
}

int current_shard_id() {
  return active_kernel->shard_id();
}

Kernel* get_kernel() {
  return active_kernel;
}

std::string get_kernel_code() {
  return active_kernel->code();
}

TableT<RefPtr, RefPtr>* create_table(Master* m, PyObject* sharder, PyObject* accum) {
  Py_IncRef(sharder);
  Py_IncRef(accum);
  return m->create_table(new PySharder(), new PyAccum(), pickle(sharder),
      pickle(accum));
}

}
