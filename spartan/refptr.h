#ifndef REFPTR_H
#define REFPTR_H

#include <Python.h>
#include <boost/intrusive_ptr.hpp>

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


#endif // REFPTR_H
