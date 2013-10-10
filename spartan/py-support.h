#ifndef SPARTAN_PY_SUPPORT_H
#define SPARTAN_PY_SUPPORT_H

#include <Python.h>
#include <boost/intrusive_ptr.hpp>
#include "util/common.h"

typedef boost::intrusive_ptr<PyObject> RefPtr;

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
    //  GILHelper h;
    Py_XINCREF(p);
  }
}

static inline void intrusive_ptr_release(PyObject* p) {
  if (Py_IsInitialized()) {
    GILHelper h;
    Py_XDECREF(p);
  }
}

template<class T>
T check(T result) {
  if (PyErr_Occurred()) {
    PyErr_Print();
    spartan::print_backtrace();
    Log_fatal("Python error, aborting.");
  }
  return result;
}

// Hashing/equality for RefPtrs
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


// Get rid of annoying writable string warnings.
#define W(str) (char*)str

// Most python functions return 'new' references; we don't need
// to incref these when turning them into a RefPtr.
extern RefPtr to_ref(PyObject* o);
extern std::string repr(RefPtr p);


class Pickler {
  RefPtr cPickle;
  RefPtr cloudpickle;
  RefPtr loads;
  RefPtr cloud_dump;
  RefPtr cpickle_dump;
  RefPtr cStringIO;
  RefPtr cStringIO_stringIO;

public:
  Pickler();

  RefPtr load(const RefPtr& py_str);
  RefPtr load(const std::string& data);
  std::string store(const RefPtr& p);
};

Pickler& get_pickler();

#endif // SPARTAN_PY_SUPPORT_H
