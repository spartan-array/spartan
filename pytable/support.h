#ifndef PYTHON_SUPPORT_H
#define PYTHON_SUPPORT_H

#include <boost/smart_ptr.hpp>
#include <Python.h>

#include "sparrow/table.h"

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
  return PyObject_Compare(a.get(), b.get()) == 0;
}
#endif

namespace sparrow {

class Kernel;
class Master;

typedef TableT<RefPtr, RefPtr> PyTable;

Master* init(int argc, char* argv[]);
PyTable* create_table(Master*, PyObject* sharder, PyObject* accum);
void foreach_shard(Master* m, Table* t, PyObject* fn, PyObject* args);

PyTable* get_table(Kernel* k, int id);

// This is a round-about way of getting SWIG to wrap some function
// arguments for us.
Kernel* as_kernel(long ptr);

} // namespace sparrow

#endif // PYTHON_SUPPORT_H
