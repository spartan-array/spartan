#include "sparrow/python/support.h"

#include "sparrow/table.h"
#include "sparrow/master.h"
#include "sparrow/kernel.h"
#include "sparrow/worker.h"

#include <Python.h>

namespace sparrow {

class PythonKernel;

PythonKernel* active_kernel;

class PythonKernel: public Kernel {
public:

  std::string code() {
    return args()["map_fn"];
  }

  void run() {
    Table* t = get_table(table_id());
    Shard* s = t->shard(shard_id());

    active_kernel = this;
    PyRun_SimpleString("import sparrow; sparrow._bootstrap_kernel()");
  }
};
REGISTER_KERNEL(PythonKernel);


void map_shards(Master* m, Table* t, const std::string& fn) {
  sparrow::RunDescriptor r;
  r.kernel = "PythonKernel";
  r.args["map_fn"] = fn;
  r.table = t;
  r.shards = sparrow::range(0, t->num_shards());
  m->run(r);
}

PyObject* get(Table* t, const TableKey& k) {
  const TableValue& v = t->get(k);
  return PyBuffer_FromMemory((void*) v.data(), v.size());
}

Master* init(int argc, char* argv[]) {
  Init(argc, argv);
  if (!StartWorker()) {
    return new Master();
  }
  return NULL;
}

Table* get_table(int id) {
  return active_kernel->get_table(id);
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

}
