#include "spartan/py-support.h"

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
  return std::string(PyString_AsString(PyObject_Repr(p.get())));
}

Pickler::Pickler() {
  GILHelper lock;
  cStringIO = to_ref(PyImport_ImportModule("cStringIO"));
  cPickle = to_ref(PyImport_ImportModule("cPickle"));
  cloudpickle = to_ref(PyImport_ImportModule("spartan.cloudpickle"));
  loads = to_ref(PyObject_GetAttrString(cPickle.get(), "loads"));
  cStringIO_stringIO = to_ref(
      PyObject_GetAttrString(cStringIO.get(), "StringIO"));
  cpickle_dump = to_ref(PyObject_GetAttrString(cPickle.get(), "dump"));
  cloud_dump = to_ref(PyObject_GetAttrString(cloudpickle.get(), "dump"));
}

RefPtr Pickler::load(const RefPtr& py_str) {
  GILHelper lock;
  //Log_info("Loading %d bytes", PyString_Size(py_str.get()));
  return to_ref(PyObject_CallFunction(loads.get(), W("O"), py_str.get()));
}

RefPtr Pickler::load(const std::string& data) {
  GILHelper lock;
  RefPtr py_str = to_ref(PyString_FromStringAndSize(data.data(), data.size()));
  return load(py_str);
}

std::string Pickler::store(const RefPtr& p) {
  GILHelper lock;
  auto out = to_ref(PyObject_CallFunction(cStringIO_stringIO.get(), W("")));
  auto result = PyObject_CallFunction(cpickle_dump.get(), W("OOi"), p.get(),
      out.get(), -1);
  if (result != NULL) {
    auto c_out = (IOobject*) out.get();
    return std::string(c_out->buf, c_out->pos);
  }

  PyErr_Clear();
  out = to_ref(PyObject_CallFunction(cStringIO_stringIO.get(), W("")));
  check(
      PyObject_CallFunction(cloud_dump.get(), W("OOi"), p.get(), out.get(),
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
