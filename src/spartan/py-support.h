#ifndef SPARTAN_PY_SUPPORT_H
#define SPARTAN_PY_SUPPORT_H

#include <boost/noncopyable.hpp>

#include "spartan/table.h"
#include "spartan/master.h"
#include "spartan/kernel.h"
#include "spartan/worker.h"
#include "spartan/refptr.h"

#include "spartan/util/common.h"

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

// Get rid of annoying writable string warnings.
#define W(str) (char*)str

// Most python functions return 'new' references; we don't need
// to incref these when turning them into a RefPtr.
extern RefPtr to_ref(PyObject* o);
extern std::string repr(RefPtr p);
extern std::string format_exc(const PyException* p);

class Pickler {
  RefPtr _cPickle;
  RefPtr _cloudpickle;
  RefPtr _load;
  RefPtr _loads;
  RefPtr _cloud_dump;
  RefPtr _cpickle_dump;
  RefPtr _cStringIO;
  RefPtr _cStringIO_stringIO;

public:
  Pickler();

  RefPtr load(const RefPtr& py_str);
  RefPtr load(const std::string& data);
  std::string store(const RefPtr& p);
};

Pickler& get_pickler();

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

namespace spartan {

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
    try {
      RefPtr result = to_ref(
          PyObject_CallFunction(code.get(), W("OOO"), k.get(), v->get(),
              update.get()));
      CHECK(result.get() != Py_None);
      CHECK(result.get() != NULL);

      *v = result;
    } catch (PyException* e) {
      Log_warn("Error during accumulate.");
      throw e;
    }
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
    const std::string& fn_pickle = args().find("map_fn")->second;
    RefPtr fn = get_pickler().load(fn_pickle);
    to_ref(PyObject_CallFunction(fn.get(), W("l"), this));
  }
  DECLARE_REGISTRY_HELPER(Kernel, PyKernel);
};

} // namespace spartan

#endif // SPARTAN_PY_SUPPORT_H
