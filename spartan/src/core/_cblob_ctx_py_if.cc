#include <Python.h>
#include <structmember.h>
//#define PY_ARRAY_UNIQUE_SYMBOL spartan_ctile_ARRAY_API
//#define NO_IMPORT_ARRAY
#include <numpy/arrayobject.h>
#include <iostream>
#include "cblob_ctx.h"
#include "array/_ctile_py_if.h"

#define RETURN_IF_NULL(val) \
do {\
    if (val == NULL) { \
        return NULL; \
    } \
} while (0)

#define RETURN_CHECK(val) \
do {\
    if (val == NULL) { \
        return NULL; \
    } else { \
        return val; \
    } \
} while (0)

class GILHelper {
    PyGILState_STATE gil_state;
public:
    GILHelper() {
        gil_state = PyGILState_Ensure();
    }

    ~GILHelper() {
        PyGILState_Release(gil_state);
    }
};

typedef struct {
    PyObject_HEAD
    class CBlobCtx *ctx;
    bool is_master;
    std::unordered_map<int32_t, spartan::WorkerProxy*> workers;

} CBlobCtx_Py;

//static bool
//is_integer(PyObject *o) {
    //if (PyLong_Check(o) || PyInt_Check(o)) {
        //return true;
    //}
    //return false;
//}

static long long
get_longlong(PyObject *o) {
    if (PyNumber_Check(o)) {
        PyObject *_long;
        long long ret;

        _long = PyNumber_Long(o);
        ret = PyLong_AsLongLong(_long);
        Py_DECREF(_long);
        return ret;
    } else {
        assert(0);
        return 0;
    }
}

static PyObject*
_CBlobCtx_Py_get(PyObject* o, PyObject *args, bool is_flatten)
{
    CBlobCtx_Py *self = (CBlobCtx_Py*) o;
    PyObject *tile_id, *subslice, *id_worker, *id_id;
    assert(self->ctx != NULL);

    if (!PyArg_ParseTuple(args, "OO", &tile_id, &subslice))
        return NULL;

    id_worker = PyObject_GetAttrString(tile_id, "worker");
    id_id = PyObject_GetAttrString(tile_id, "id");
    assert(id_worker != NULL);
    assert(id_id != NULL);
    TileId id(get_longlong(id_worker), get_longlong(id_id));
    CSliceIdx idx(subslice, 0, NULL);

    GetResp *resp = new GetResp();
    rpc::Future *fu = NULL;
    Py_BEGIN_ALLOW_THREADS
        fu = self->ctx->py_get(&id, &idx, resp);
    Py_END_ALLOW_THREADS
    PyObject *mod = PyImport_ImportModule("spartan.rpc.future");
    PyObject *obj = PyObject_GetAttrString(mod, "Future_Get");
    if (fu == NULL) {
        if (is_flatten)
            return PyObject_CallFunction(obj, (char*)"(OkO)", Py_None,
                                         (unsigned long)resp, Py_True, NULL);
        else
            return PyObject_CallFunction(obj, (char*)"(OkO)", Py_None,
                                         (unsigned long)resp, Py_False, NULL);
    } else {
        delete resp;
        if (is_flatten)
            return PyObject_CallFunction(obj, (char*)"(kOO)", (unsigned long)fu,
                                         Py_None, Py_True);
        else
            return PyObject_CallFunction(obj, (char*)"(kOO)", (unsigned long)fu,
                                         Py_None, Py_False);
    }
}

static PyObject *
CBlobCtx_Py_get(PyObject* o, PyObject *args)
{
    return _CBlobCtx_Py_get(o, args, false);
}

static PyObject *
CBlobCtx_Py_get_flatten(PyObject* o, PyObject *args)
{
    return _CBlobCtx_Py_get(o, args, true);
}

static PyObject *
CBlobCtx_Py_update(PyObject* o, PyObject *args)
{
    PyObject *tile_id, *subslice, *id_worker, *id_id;
    CBlobCtx_Py *self = (CBlobCtx_Py*) o;
    unsigned long reducer, ctile_u;

    assert(self->ctx != NULL);

    if (!PyArg_ParseTuple(args, "OOkk", &tile_id, &subslice, &ctile_u, &reducer))
        return NULL;

    CTile* tile = (CTile*) ctile_u;
    id_worker = PyObject_GetAttrString(tile_id, "worker");
    id_id = PyObject_GetAttrString(tile_id, "id");
    assert(id_worker != NULL);
    assert(id_id != NULL);
    TileId id(get_longlong(id_worker), get_longlong(id_id));
    CSliceIdx idx(subslice, 0, NULL);

    rpc::Future *fu = NULL;
    Py_BEGIN_ALLOW_THREADS
        fu = self->ctx->py_update(&id, &idx, tile, reducer);
    Py_END_ALLOW_THREADS

    PyObject *mod = PyImport_ImportModule("spartan.rpc.future");
    PyObject *obj = PyObject_GetAttrString(mod, "Future");
    if (fu == NULL) {
        PyObject *core_mod = PyImport_ImportModule("spartan.core");
        PyObject *core_obj = PyObject_GetAttrString(core_mod, "EmptyMessage");
        PyObject *resp = PyObject_CallFunction(core_obj, NULL);
        PyObject *kargs = PyDict_New();
        PyDict_SetItemString(kargs, "id", PyInt_FromLong(-1));
        PyDict_SetItemString(kargs, "resp", resp);
        return PyObject_Call(obj, PyTuple_New(0), kargs);
    } else {
        PyObject *rep_type = Py_BuildValue("(s)", "EmptyMessage");
        PyObject *kargs = PyDict_New();
        PyDict_SetItemString(kargs, "id", Py_BuildValue("k", (unsigned long)fu));
        PyDict_SetItemString(kargs, "rep_type", rep_type);
        return PyObject_Call(obj, PyTuple_New(0), kargs);
    }
}

static PyObject *
CBlobCtx_Py_create(PyObject* o, PyObject *args)
{
    CBlobCtx_Py *self = (CBlobCtx_Py*) o;
    PyObject *tile, *tile_id, *id_worker, *id_id;
    assert(self->ctx != NULL);

    if (!PyArg_ParseTuple(args, "OOO", &tile_id, &tile))
        return NULL;

    id_worker = PyObject_GetAttrString(tile_id, "worker");
    id_id = PyObject_GetAttrString(tile_id, "id");
    assert(id_worker != NULL);
    assert(id_id != NULL);
    TileId id(get_longlong(id_worker), get_longlong(id_id));

    TileIdMessage resp;
    rpc::Future *fu = NULL;
    Py_BEGIN_ALLOW_THREADS
        fu = self->ctx->py_create(((TileBase*)tile)->c_tile, &id, &resp);
    Py_END_ALLOW_THREADS

    PyObject *mod = PyImport_ImportModule("spartan.rpc.future");
    PyObject *obj = PyObject_GetAttrString(mod, "Future");
    if (fu == NULL) {
        PyObject *core_mod = PyImport_ImportModule("spartan.core");
        PyObject *core_obj = PyObject_GetAttrString(core_mod, "TileId");
        PyObject *tileid = PyObject_CallFunction(core_obj, (char*)"kk",
                                                 resp.tile_id.worker,
                                                 resp.tile_id.id);
        core_obj = PyObject_GetAttrString(mod, "TileIdMessage");
        PyObject *message = PyObject_CallFunctionObjArgs(core_obj, tileid, NULL);
        mod = PyImport_ImportModule("spartan.rpc.future");
        obj = PyObject_GetAttrString(mod, "Future");
        PyObject *kargs = PyDict_New();
        PyDict_SetItemString(kargs, "id", PyInt_FromLong(-1));
        PyDict_SetItemString(kargs, "rep", message);
        return PyObject_Call(obj, PyTuple_New(0), kargs);
    } else {
        PyObject *rep_type = Py_BuildValue("(s)", "TileIdMessage");
        PyObject *kargs = PyDict_New();
        PyDict_SetItemString(kargs, "id", Py_BuildValue("k", (unsigned long)fu));
        PyDict_SetItemString(kargs, "rep_type", rep_type);
        return PyObject_Call(obj, PyTuple_New(0), kargs);
    }
}

static void
CBlobCtx_Py_delete_workers(PyObject *o, PyObject *args)
{
    /* TODO: */
    assert(false);
}

static int
CBlobCtx_Py_init(PyObject *o, PyObject *args, PyObject *kwds)
{
    PyObject *workers;
    unsigned long worker_id, ctx_u;
    CBlobCtx_Py *self = (CBlobCtx_Py*)o;

    if (!PyArg_ParseTuple(args, "kOk", &worker_id, &workers, &ctx_u))
        return -1;

    assert (PyDict_Check(workers));

    PyObject *key, *value;
    Py_ssize_t pos = 0;
    self->workers.clear();
    while (PyDict_Next(workers, &pos, &key, &value)) {
        rpc::Client *client = (rpc::Client*) get_longlong(value);
        self->workers[get_longlong(key)] = new spartan::WorkerProxy(client);
    }

    if (ctx_u == 0) { // This is the master!
        self->ctx = new CBlobCtx(worker_id, &self->workers, NULL);
        self->is_master = true;
    } else {
        self->ctx = (CBlobCtx*) ctx_u;
        self->is_master = false;
    }
    return 0;
}

static void
CBlobCtx_Py_dealloc(PyObject *o)
{
    CBlobCtx_Py *self = (CBlobCtx_Py*) o;

    if (self->is_master) {
        for (auto it = self->workers.begin(); it != self->workers.end(); ++it) {
            delete it->second;
        }
    }
}

static PyMemberDef CBlobCtx_Py_members[] = {
    {NULL} /* Sentinel */
};

static PyGetSetDef CBlobCtx_Py_getseters[] = {
    {NULL} /* Sentinel */
};

static PyMethodDef CBlobCtx_Py_methods[] = {
    {"delete_workers", (PyCFunction)CBlobCtx_Py_delete_workers, METH_VARARGS,
     "Deleter workers from CBlobCtx (should not be called from workers)"},
    {"get", (PyCFunction)CBlobCtx_Py_get, METH_VARARGS,
     "get RPC."},
    {"get_flatten", (PyCFunction)CBlobCtx_Py_get_flatten, METH_VARARGS,
     "get_flatten RPC."},
    {"update", (PyCFunction)CBlobCtx_Py_update, METH_VARARGS,
     "update RPC."},
    {"create", (PyCFunction)CBlobCtx_Py_create, METH_NOARGS,
     "create RPC."},
    {NULL} /* Sentinel */
};

static PyTypeObject CBlobCtx_PyType = {
    PyObject_HEAD_INIT(NULL)
    0,                           /* ob_size */
    "blob_ctx.CBlobCtx_Py",     /* tp_name */
    sizeof(CBlobCtx_Py),        /* tp_basicsize */
    0,                           /* tp_itemsize */
    CBlobCtx_Py_dealloc,        /* tp_dealloc */
    0,                           /* tp_print */
    0,                           /* tp_getattr */
    0,                           /* tp_setattr */
    0,                           /* tp_compare */
    0,                           /* tp_repr */
    0,                           /* tp_as_number */
    0,                           /* tp_as_sequence */
    0,                           /* tp_as_mapping */
    0,                           /* tp_hash */
    0,                           /* tp_call */
    0,                           /* tp_str */
    0,                           /* tp_getattro */
    0,                           /* tp_setattro */
    0,                           /* tp_as_buffer */
    Py_TPFLAGS_DEFAULT,          /* tp_flags */
    "blob_ctx base object",      /* tp_doc */
    0,                           /* tp_traverse */
    0,                           /* tp_clear */
    0,                           /* tp_richcompare */
    0,                           /* tp_weaklistoffset */
    0,                           /* tp_iter */
    0,                           /* tp_iternext */
    CBlobCtx_Py_methods,         /* tp_methods */
    CBlobCtx_Py_members,         /* tp_members */
    CBlobCtx_Py_getseters,       /* tp_getset */
    0,                           /* tp_base */
    0,                           /* tp_dict */
    0,                           /* tp_descr_get */
    0,                           /* tp_descr_set */
    0,                           /* tp_dictoffset */
    CBlobCtx_Py_init,            /* tp_init */
    0,                           /* tp_alloc */
    0,                           /* tp_new */
};

static PyMethodDef _cblob_ctx_py_if_methods[] = {
    {NULL}  /* Sentinel */
};

#ifndef PyMODINIT_FUNC  /* declarations for DLL import/export */
#define PyMODINIT_FUNC void
#endif
PyMODINIT_FUNC
init_cblob_ctx_py_if(void)
{
    PyObject* m;

    CBlobCtx_PyType.tp_new = PyType_GenericNew;
    if (PyType_Ready(&CBlobCtx_PyType) < 0)
        return;

    m = Py_InitModule3("_cblob_ctx_if", _cblob_ctx_py_if_methods,
                       "Python interface for cblob_ctx module");

    Py_INCREF(&CBlobCtx_PyType);
    PyModule_AddObject(m, "CBlobCtx_Py", (PyObject *)&CBlobCtx_PyType);
}

