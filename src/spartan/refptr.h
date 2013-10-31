#ifndef REFPTR_H
#define REFPTR_H

#include <Python.h>
#include <boost/intrusive_ptr.hpp>
#include <boost/noncopyable.hpp>

struct PyException : private boost::noncopyable {
  PyException();
  PyException(std::string value_str);

  PyObject* traceback;
  PyObject* value;
  PyObject* type;
};

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

template<class T>
T check(T result) {
  if (PyErr_Occurred()) {
    throw new PyException;
  }
  return result;
}

typedef boost::intrusive_ptr<PyObject> RefPtr;

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
    auto gstate = PyGILState_Ensure();
    Py_XDECREF(p);
    PyGILState_Release(gstate);
  }
}

static inline bool operator==(const RefPtr& a, const RefPtr& b) {
  GILHelper lock;
  return check(PyObject_RichCompare(a.get(), b.get(), Py_EQ));
}

// Hashing/equality for RefPtrs
namespace boost {
static inline size_t hash_value(const RefPtr& p) {
  GILHelper lock;
  return check(PyObject_Hash(p.get()));
}
} // namespace boost


#endif // REFPTR_H
