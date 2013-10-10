#ifndef SPARTAN_SUPPORT_H
#define SPARTAN_SUPPORT_H

#include "spartan/table.h"
#include "spartan/master.h"
#include "spartan/kernel.h"
#include "spartan/worker.h"

#include "spartan/util/common.h"

#include "Python.h"
#include <boost/intrusive_ptr.hpp>
#include <string>

// Various helper methods for pickling/unpickling, managing
// Python pointers, etc.



template<class T>
static inline PyObject* get_code(T* obj) {
  if (obj == NULL) {
    Py_INCREF(Py_None);
    return Py_None;
  }

  RefPtr p = obj->code;
  Py_INCREF(p.get());
  return p.get();
}

namespace spartan {

template<class T>
class PyInitableT: public T {
public:
  RefPtr code;
  std::string opts_;

  void init(const std::string& opts) {
    opts_ = opts;
    code = get_pickler().load(opts);
  }

  PyInitableT() {
  }
  PyInitableT(const std::string& opts) {
    init(opts);
  }

  const std::string& opts() {
    return opts_;
  }
};

class PySharder: public PyInitableT<Sharder> {
public:
  size_t shard_for_key(const RefPtr& k, int num_shards) const {
    GILHelper lock;
    RefPtr result = to_ref(
        PyObject_CallFunction(code.get(), W("Oi"), k.get(), num_shards));

    CHECK(PyInt_Check(result.get()));
    return PyInt_AsLong(result.get());
  }
  DECLARE_REGISTRY_HELPER(Sharder, PySharder);
};

class PyAccum: public PyInitableT<Accumulator> {
public:
  void accumulate(const RefPtr& k, RefPtr* v, const RefPtr& update) const {
    GILHelper lock;
    RefPtr result = to_ref(
        PyObject_CallFunction(code.get(), W("OOO"), k.get(), v->get(),
            update.get()));
    CHECK(result.get() != Py_None);
    CHECK(result.get() != NULL);

    *v = result;
  }
  DECLARE_REGISTRY_HELPER(Accumulator, PyAccum);
};

class PySelector: public PyInitableT<Selector> {
  RefPtr select(const RefPtr& k, const RefPtr& v) {
    GILHelper lock;

    RefPtr result = to_ref(
        PyObject_CallFunction(code.get(), W("OO"), k.get(), v.get()));
    return result;
  }

  DECLARE_REGISTRY_HELPER(Selector, PySelector);
};

class PyKernel: public Kernel {
public:
  void run() {
    GILHelper lock;
    RefPtr fn = get_pickler().load(args()["map_fn"]);
    to_ref(PyObject_CallFunction(fn.get(), W("l"), this));
  }
  DECLARE_REGISTRY_HELPER(Kernel, PyKernel);
};

} // namespace spartan

#endif // SPARTAN_SUPPORT_H
