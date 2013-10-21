#include "spartan/py-support.h"

#include "spartan/table.h"
#include "spartan/master.h"
#include "spartan/kernel.h"
#include "spartan/worker.h"

#include "spartan/util/common.h"

#include "Python.h"
#include <boost/intrusive_ptr.hpp>
#include <string>

typedef struct {
  PyObject_HEAD
  char *buf;
  Py_ssize_t pos, string_size;
} IOobject;

// Most python functions return 'new' references; we don't need
// to incref these when turning them into a RefPtr.
RefPtr to_ref(PyObject* o) {
  return RefPtr(check(o), false);
}

static Pickler* _pickler = NULL;
Pickler& get_pickler() {
  if (_pickler == NULL) {
    _pickler = new Pickler;
  }
  return *_pickler;
}

std::string repr(RefPtr p) {
  GILHelper lock;
  return std::string(PyString_AsString(PyObject_Repr(p.get())));
}

Pickler::Pickler() {
  GILHelper lock;
  _cStringIO = to_ref(PyImport_ImportModule("cStringIO"));
  _cPickle = to_ref(PyImport_ImportModule("cPickle"));
  _cloudpickle = to_ref(PyImport_ImportModule("spartan.cloudpickle"));
  _loads = to_ref(PyObject_GetAttrString(_cPickle.get(), "loads"));
  _load = to_ref(PyObject_GetAttrString(_cPickle.get(), "load"));
  _cStringIO_stringIO = to_ref(
      PyObject_GetAttrString(_cStringIO.get(), "StringIO"));
  _cpickle_dump = to_ref(PyObject_GetAttrString(_cPickle.get(), "dump"));
  _cloud_dump = to_ref(PyObject_GetAttrString(_cloudpickle.get(), "dump"));
}

RefPtr Pickler::load(const RefPtr& py_str) {
  GILHelper lock;
  //Log_info("Loading %d bytes", PyString_Size(py_str.get()));
  return to_ref(PyObject_CallFunction(_loads.get(), W("O"), py_str.get()));
}

RefPtr Pickler::load(const std::string& data) {
  GILHelper lock;
  try {
    RefPtr buffer = to_ref(PyBuffer_FromMemory((void*) data.data(), data.size()));
    auto in = to_ref(
        PyObject_CallFunction(_cStringIO_stringIO.get(), W("O"), buffer.get()));
    return to_ref(PyObject_CallFunction(_load.get(), W("O"), in.get()));
  } catch(PyException* exc) {
    Log_info("Exception while loading pickle.");
    throw exc;
  }
}

std::string Pickler::store(const RefPtr& p) {
  GILHelper lock;
  auto out = to_ref(PyObject_CallFunction(_cStringIO_stringIO.get(), W("")));
  auto result = PyObject_CallFunction(_cpickle_dump.get(), W("OOi"), p.get(),
      out.get(), -1);
  if (result != NULL) {
    auto c_out = (IOobject*) out.get();
    return std::string(c_out->buf, c_out->pos);
  }

  PyErr_Clear();
  out = to_ref(PyObject_CallFunction(_cStringIO_stringIO.get(), W("")));
  check(
      PyObject_CallFunction(_cloud_dump.get(), W("OOi"), p.get(), out.get(),
          -1));
  auto c_out = (IOobject*) out.get();
  return std::string(c_out->buf, c_out->pos);
}

rpc::Marshal& operator <<(rpc::Marshal& m, const RefPtr& p) {
  m << get_pickler().store(p);
  return m;
}

rpc::Marshal& operator >>(rpc::Marshal& m, RefPtr& p) {
  std::string s;
  m >> s;
  auto r = get_pickler().load(s);
  p.swap(r);
  return m;
}

std::string format_exc(const PyException* p) {
  GILHelper gil;
  RefPtr tb = to_ref(PyImport_ImportModule("traceback"));
  RefPtr list = to_ref(
      PyObject_CallMethod(tb.get(), W("format_exception"), W("OOO"),
          p->type,
          p->value ? p->value : Py_None,
          p->traceback ? p->traceback : Py_None));
  RefPtr empty = to_ref(PyUnicode_FromString(""));
  RefPtr pystr = to_ref(PyUnicode_Join(empty.get(), list.get()));
  return std::string(PyString_AsString(PyUnicode_AsLatin1String(pystr.get())));
}

PyException::PyException() {
  //Log_warn("Exceptin!");
  GILHelper gil;
  CHECK_NE(PyErr_Occurred(), NULL);
  PyErr_Fetch(&type, &value, &traceback);
  CHECK_NE(type, NULL);
  Log_warn("%s", format_exc(this).c_str());
}

PyException::PyException(std::string value_str) {
  //Log_warn("Exceptin!");
  GILHelper gil;
  PyErr_SetString(PyExc_SystemError, value_str.c_str());
  PyErr_Fetch(&type, &value, &traceback);
}

namespace spartan {

DEFINE_REGISTRY_HELPER(Sharder, PySharder);
DEFINE_REGISTRY_HELPER(Accumulator, PyAccum);
DEFINE_REGISTRY_HELPER(Selector, PySelector);
DEFINE_REGISTRY_HELPER(Kernel, PyKernel);

} // namespace spartan
