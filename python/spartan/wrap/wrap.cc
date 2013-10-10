#include "spartan/wrap/wrap.h"

namespace spartan {

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


DEFINE_REGISTRY_HELPER(Sharder, PySharder);
DEFINE_REGISTRY_HELPER(Accumulator, PyAccum);
DEFINE_REGISTRY_HELPER(Selector, PySelector);
DEFINE_REGISTRY_HELPER(Kernel, PyKernel);

}  // namespace spartan
