#include <Python.h>
#include <structmember.h>
//#define PY_ARRAY_UNIQUE_SYMBOL spartan_ctile_ARRAY_API
//#define NO_IMPORT_ARRAY
#include <numpy/arrayobject.h>
#include <iostream>

#include "rpc/server.h"
#include "rpc/client.h"
#include "array/ctile.h"
#include "core/ccore.h"

static PyObject*
numpy_to_ctile(PyObject* o, PyObject *args)
{
    CTile *tile; 

    tile = ctile_creator(args);
    assert(tile != NULL);
    return Py_BuildValue("k", (unsigned long)tile);
}

static void
release_ctile(PyObject* o, PyObject *args)
{
    unsigned long u;
    if (!PyArg_ParseTuple(args, "k", &u))
        return NULL;

    delete (CTile*)u;
}

static PyObject *
deserialize_get_resp(PyObject* o, PyObject *args)
{
    unsigned long u;
    if (!PyArg_ParseTuple(args, "k", &u))
        return NULL;
    
    Marshal* m = (Marshal *) u;
    GetResp* resp = new GetResp();
    *m >> *resp;

    return Py_BuildValue("k", (unsigned long)resp);
}

static PyObject *
get_resp_to_tile(PyObject* o, PyObject *args)
{
    unsigned long u; 
    if (!PyArg_ParseTuple(args, "k", &u)) 
        return NULL;
    
    GetResp* resp = (GetResp*) u;
    assert(!resp->own_data);

    //PyObject *mod = PyImport_ImportModule("spartan.array.tile");
    //PyObject *obj = PyObject_GetAttrString(mod, "Tile");
    //PyObject *args = Py_BuildValue("(OOOOO)", Py_None, Py_None, Py_None, 
                                              //Py_None, Py_None);
    //PyObject *kargs = PyDict_New();
    //PyDict_SetItem(kargs, PyString_FromString("private_id"),
                   //Py_BuildValue("k", (unsigned long)ctile));
    //PyObject *tile = PyObject_Call(obj, args, kargs);
    
    CTile_RPC *rpc = vector_to_ctile_rpc(resp->data);
    CTile *ctile = new CTile(rpc);
    release_ctile_rpc(rpc);
    PyObject *obj = ctile->to_npy();
    delete ctile;

    return obj;
}

static PyMethodDef _rpc_ctile_methods[] = {
    {"numpy_to_ctile", numpy_to_ctile, METH_VARARGS, ""},
    {"release_ctile", release_ctile, METH_VARAGS, ""},
    {"deserialize_get_resp", deserialize_get_Resp, METH_VARARGS, ""},
    {"get_resp_to_tile", get_resp_to_tile, METH_VARARGS, ""},
    {NULL}  /* Sentinel */
};

#ifndef PyMODINIT_FUNC  /* declarations for DLL import/export */
#define PyMODINIT_FUNC void
#endif
PyMODINIT_FUNC
init_rpc_ctile_py_if(void) 
{
    PyObject* m;

    m = Py_InitModule3("_rpc_ctile", _rpc_ctile_methods,
                       "Python interface for rpc_ctile module");
}
