#ifndef PYTHON_SUPPORT_H
#define PYTHON_SUPPORT_H

#include "sparrow/master.h"
#include "sparrow/worker.h"
#include <boost/smart_ptr.hpp>

#include <Python.h>

#ifndef SWIG
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
  return PyObject_Compare(a.get(), b.get());
}
#endif

namespace sparrow {
typedef TableT<RefPtr, RefPtr> PyTable;

PyTable* create_table(Master*, PyObject* sharder, PyObject* accum);

void _map_shards(Master* m, Table* t, const std::string& fn,
    const std::string& args);

Master* init(int argc, char* argv[]);

Kernel* get_kernel();
PyTable* get_table(int id);
int current_table_id();
int current_shard_id();

} // namespace sparrow

#endif // PYTHON_SUPPORT_H
