#include <Python.h>
#include <structmember.h>
#define PY_ARRAY_UNIQUE_SYMBOL spartan_ctile_ARRAY_API
#define NO_IMPORT_ARRAY
#include <numpy/arrayobject.h>
#include <iostream>
#include "ctile.h"
#include "_ctile_py_if.h"

#define RETURN_IF_ERROR(expr) \
do {\
    int ret; \
    ret = (expr); \
    if (ret == -1) { \
        return; \
    } \
} while (0)

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
    }
    return 0;
}

static void
TileBase_dealloc(PyObject *o)
{
    TileBase *self = (TileBase*) o;
    Log_debug("%s, tile = %p, ctile = %p", __func__, o, self->c_tile);
    self->c_tile->decrease_py_c_refcount();
    if (self->c_tile->can_release()) {
        Log_debug("%s, can release the ctile %p", self->c_tile);
        delete self->c_tile;
    } else {
        Log_debug("%s, can't release the ctile %p", self->c_tile);
    }
    self->c_tile = NULL;
    Py_DECREF(self->shape);
    Py_DECREF(self->dtype);
    self->ob_type->tp_free((PyObject*)self);
}

//static PyObject *
//TileBase_new(PyTypeObject *type, PyObject *args, PyObject *kwds)
//{
    //TileBase *self;

    //self = (TileBase*)type->tp_alloc(type, 0);
    //if (self != NULL) {
        ////self->c_tile = new CTile();
        //self->c_tile = NULL;
    //}

    //return (PyObject *) self;
//}

static PyObject*
TileBase_reduce(PyObject* o)
{
    TileBase *self = (TileBase*) o;
    PyObject *mod, *result, *tuple, *obj;
    PyObject *shape, *dtype, *tile_type, *sparse_type;

    Log_debug("%s", __func__);
    result = PyTuple_New(2);
    tuple = PyTuple_New(4);
    char dtype_str[] = {self->c_tile->get_dtype(), '\0'};
    dtype = PyString_FromString(dtype_str);
    tile_type = PyInt_FromLong((long)self->c_tile->get_type());
    sparse_type = PyInt_FromLong((long)self->c_tile->get_sparse_type());
    shape = PyTuple_New(self->c_tile->get_nd());
    for (int i = 0; i < self->c_tile->get_nd(); i++) {
        PyTuple_SetItem(shape, i, PyLong_FromLongLong(self->c_tile->get_dimensions()[i]));
    }

    mod = PyImport_ImportModule("spartan.array.tile");
    obj = PyObject_GetAttrString(mod, "from_shape");
    Py_DECREF(mod);
    PyTuple_SetItem(tuple, 0, shape);
    PyTuple_SetItem(tuple, 1, dtype);
    PyTuple_SetItem(tuple, 2, tile_type);
    PyTuple_SetItem(tuple, 3,  sparse_type);
    PyTuple_SetItem(result, 0,  obj);
    PyTuple_SetItem(result, 1,  tuple);
    return result;
}

static PyObject *
TileBase_get(PyObject* o, PyObject* args)
{
    TileBase *self = (TileBase*)o;
    PyObject *slice;

    Log_debug("%s", __func__);

    if (!PyArg_ParseTuple(args, "O",  &slice))
        return NULL;

    CSliceIdx cslice_idx(slice, self->c_tile->get_nd(), self->c_tile->get_dimensions());

    std::vector<char*> rpc_vector = self->c_tile->get(cslice_idx);
    CTile_RPC *rpc = (CTile_RPC*) vector_to_ctile_rpc(rpc_vector);
    CTile *ctile = new CTile(rpc);
    PyObject *ret = ctile->to_npy();

    if (ret == NULL) {
        delete(rpc);
    }
    return ret;
}

static PyObject *
TileBase__update(PyObject* o, PyObject* args)
{
    TileBase *self = (TileBase*)o;
    PyObject *slice, *data, *tile_type, *sparse_type, *reducer;

    Log_debug("%s", __func__);
    if (!PyArg_ParseTuple(args, "OOOOO", &slice, &tile_type, &sparse_type, &data, &reducer))
        return NULL;

    CSliceIdx cslice_idx(slice, self->c_tile->get_nd(), self->c_tile->get_dimensions());
    CTILE_TYPE ttype = (CTILE_TYPE)get_longlong(tile_type);
    CTILE_SPARSE_TYPE stype = (CTILE_SPARSE_TYPE)get_longlong(sparse_type);
    npy_intp npy_reducer;
    if (PyInt_Check(reducer)) {
        npy_reducer = (npy_intp)get_longlong(reducer);
    } else {
        npy_reducer = (npy_intp)reducer;
    }

    if (ttype != CTILE_SPARSE) {
        /* TODO:Release dense*/
        PyArrayObject *dense = (PyArrayObject*)PyTuple_GetItem(data, 0);
        dense = PyArray_GETCONTIGUOUS(dense);
        CTile tile(dense->dimensions, dense->nd, dense->descr->type, ttype, CTILE_SPARSE_NONE);
        CArray *dense_array = new CArray(dense->dimensions, dense->nd,
                                         dense->descr->type, dense->data,
                                         dense);
        CArray *mask_array = NULL;
        if (ttype == CTILE_MASKED) {
            PyArrayObject *mask = (PyArrayObject*)PyTuple_GetItem(data, 1);
            mask_array = new CArray(mask->dimensions, mask->nd,
                                    mask->descr->type, mask->data,
                                    mask);
        }
        tile.set_data(dense_array, mask_array);
        self->c_tile->update(cslice_idx, tile, npy_reducer);
    } else {
        CArray *sparse_array[3];
        for (int i = 0; i < 3; i++) {
            PyArrayObject *sparse = (PyArrayObject*)PyTuple_GetItem(data, i);
            sparse_array[i] = new CArray(sparse->dimensions, sparse->nd,
                                         sparse->descr->type, sparse->data,
                                         sparse);
        }
        CExtent *ex = from_slice(cslice_idx, self->c_tile->get_dimensions(),
                                 self->c_tile->get_nd());
        CTile tile(ex->shape, sparse_array[2]->get_nd(), sparse_array[2]->get_type(),
                   ttype, stype);
        tile.set_data(sparse_array);
        self->c_tile->update(cslice_idx, tile, npy_reducer);
    }

    Py_INCREF(o);
    return o;
}

static int
TileBase_init(PyObject *o, PyObject *args, PyObject *kwds)
{
    PyObject *cid = NULL;
    Log_debug("%s", __func__);
    if (kwds != Py_None) {
        assert(PyDict_Check(kwds) != 0);
        cid = PyDict_GetItemString(kwds, (char*)"ctile_id");
        assert(cid != NULL);
    }

    TileBase *self = (TileBase*)o;
    if (cid != NULL && cid != Py_None) {
        if (PyInt_Check(cid))
            self->c_tile = (CTile*) PyInt_AsLong(cid);
        else if(PyLong_Check(cid))
            self->c_tile = (CTile*) PyLong_AsLongLong(cid);
        else
            assert(0);
    } else {
        self->c_tile = ctile_creator(args);
        if (self->c_tile == NULL)
            return -1;
    }

    npy_intp *dimensions = self->c_tile->get_dimensions();
    PyObject *shape = PyTuple_New(self->c_tile->get_nd());
    for (int i = 0; i < self->c_tile->get_nd(); ++i) {
        PyTuple_SetItem(shape, i, Py_BuildValue("k", dimensions[i]));
    }
    self->shape = shape;
    int dtype_num = npy_type_token_to_number(self->c_tile->get_dtype());
    self->dtype = (PyObject*)PyArray_DescrNewFromType(dtype_num);
    self->type = self->c_tile->get_type();
    // Tell c++ part that someone else is using this CTile.
    self->c_tile->increase_py_c_refcount();
    Log_debug("%s done", __func__);
    return 0;
}

static PyObject*
TileBase_repr(PyObject *o)
{
    TileBase *self = (TileBase*)o;

    return PyString_FromFormat("%p %p", (void*)self, (void*)self->c_tile);
}

static PyObject*
TileBase_getdata(TileBase *self, void *closure)
{
    return self->c_tile->to_npy();
}

static PyMemberDef TileBase_members[] = {
    {(char*)"type", T_INT, offsetof(TileBase, type), 0, (char*)"type"},
    {(char*)"shape", T_OBJECT_EX, offsetof(TileBase, shape), 0, (char*)"shape"},
    {(char*)"dtype", T_OBJECT_EX, offsetof(TileBase, dtype), 0, (char*)"dtype"},
    {NULL} /* Sentinel */
};

static PyGetSetDef TileBase_getseters[] = {
    {(char*)"data", (getter)TileBase_getdata, NULL, (char*)"data", NULL},
    {NULL} /* Sentinel */
};

static PyMethodDef TileBase_methods[] = {
    {"__reduce__", (PyCFunction)TileBase_reduce, METH_NOARGS,
     "__reduce__"},
    {(char*)"get", (PyCFunction)TileBase_get, METH_VARARGS,
     (char*)"Get data from the tile."},
    {(char*)"_update", (PyCFunction)TileBase__update, METH_VARARGS,
     (char*)"Update data to the tile."},
    {NULL} /* Sentinel */
};

static PyTypeObject TileBaseType = {
    PyObject_HEAD_INIT(NULL)
    0,                           /* ob_size */
    "tile.TileBase",             /* tp_name */
    sizeof(TileBase),            /* tp_basicsize */
    0,                           /* tp_itemsize */
    TileBase_dealloc,            /* tp_dealloc */
    0,                           /* tp_print */
    0,                           /* tp_getattr */
    0,                           /* tp_setattr */
    0,                           /* tp_compare */
    TileBase_repr,               /* tp_repr */
    0,                           /* tp_as_number */
    0,                           /* tp_as_sequence */
    0,                           /* tp_as_mapping */
    0,                           /* tp_hash */
    0,                           /* tp_call */
    0,                           /* tp_str */
    0,                           /* tp_getattro */
    0,                           /* tp_setattro */
    0,                           /* tp_as_buffer */
    Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE,          /* tp_flags */
    "tile base object",          /* tp_doc */
    0,                           /* tp_traverse */
    0,                           /* tp_clear */
    0,                           /* tp_richcompare */
    0,                           /* tp_weaklistoffset */
    0,                           /* tp_iter */
    0,                           /* tp_iternext */
    TileBase_methods,            /* tp_methods */
    TileBase_members,            /* tp_members */
    TileBase_getseters,          /* tp_getset */
    0,                           /* tp_base */
    0,                           /* tp_dict */
    0,                           /* tp_descr_get */
    0,                           /* tp_descr_set */
    0,                           /* tp_dictoffset */
    TileBase_init,               /* tp_init */
    0,                           /* tp_alloc */
    0,                           /* tp_new */
};

static PyMethodDef extent_methods[] = {
    {NULL}  /* Sentinel */
};

#ifndef PyMODINIT_FUNC  /* declarations for DLL import/export */
#define PyMODINIT_FUNC void
#endif
PyMODINIT_FUNC
init_ctile_py_if(void)
{
    PyObject* m;

    TileBaseType.tp_new = PyType_GenericNew;
    /* Initialize class-wise members here */
    PyObject *tp_dict = PyDict_New();
    if (tp_dict == NULL) {
        return;
    }
    RETURN_IF_ERROR(PyDict_SetItem(tp_dict,
                                   PyString_FromString("TILE_DENSE"),
                                   PyInt_FromLong((long)CTILE_DENSE)));
    RETURN_IF_ERROR(PyDict_SetItem(tp_dict,
                                   PyString_FromString("TILE_MASKED"),
                                   PyInt_FromLong((long)CTILE_MASKED)));
    RETURN_IF_ERROR(PyDict_SetItem(tp_dict,
                                   PyString_FromString("TILE_SPARSE"),
                                   PyInt_FromLong((long)CTILE_SPARSE)));
    RETURN_IF_ERROR(PyDict_SetItem(tp_dict,
                                   PyString_FromString("TILE_SPARSE_NONE"),
                                   PyInt_FromLong((long)CTILE_SPARSE_NONE)));
    RETURN_IF_ERROR(PyDict_SetItem(tp_dict,
                                   PyString_FromString("TILE_SPARSE_COO"),
                                   PyInt_FromLong((long)CTILE_SPARSE_COO)));
    RETURN_IF_ERROR(PyDict_SetItem(tp_dict,
                                   PyString_FromString("TILE_SPARSE_CSC"),
                                   PyInt_FromLong((long)CTILE_SPARSE_CSC)));
    RETURN_IF_ERROR(PyDict_SetItem(tp_dict,
                                   PyString_FromString("TILE_SPARSE_CSR"),
                                   PyInt_FromLong((long)CTILE_SPARSE_CSR)));
    RETURN_IF_ERROR(PyDict_SetItem(tp_dict,
                                   PyString_FromString("TILE_REDUCER_REPLACE"),
                                   PyInt_FromLong((long)REDUCER_REPLACE)));
    RETURN_IF_ERROR(PyDict_SetItem(tp_dict,
                                   PyString_FromString("TILE_REDUCER_ADD"),
                                   PyInt_FromLong((long)REDUCER_ADD)));
    RETURN_IF_ERROR(PyDict_SetItem(tp_dict,
                                   PyString_FromString("TILE_REDUCER_MUL"),
                                   PyInt_FromLong((long)REDUCER_MUL)));
    RETURN_IF_ERROR(PyDict_SetItem(tp_dict,
                                   PyString_FromString("TILE_REDUCER_MAXIMUM"),
                                   PyInt_FromLong((long)REDUCER_MAXIMUM)));
    RETURN_IF_ERROR(PyDict_SetItem(tp_dict,
                                   PyString_FromString("TILE_REDUCER_MINIMUM"),
                                   PyInt_FromLong((long)REDUCER_MINIMUM)));
    RETURN_IF_ERROR(PyDict_SetItem(tp_dict,
                                   PyString_FromString("TILE_REDUCER_AND"),
                                   PyInt_FromLong((long)REDUCER_AND)));
    RETURN_IF_ERROR(PyDict_SetItem(tp_dict,
                                   PyString_FromString("TILE_REDUCER_XOR"),
                                   PyInt_FromLong((long)REDUCER_XOR)));
    RETURN_IF_ERROR(PyDict_SetItem(tp_dict,
                                   PyString_FromString("TILE_REDUCER_OR"),
                                   PyInt_FromLong((long)REDUCER_OR)));
    TileBaseType.tp_dict = tp_dict;
    if (PyType_Ready(&TileBaseType) < 0)
        return;

    m = Py_InitModule3("_ctile_py_if", extent_methods,
                       "Python interface for ctile module");

    Py_INCREF(&TileBaseType);
    PyModule_AddObject(m, "TileBase", (PyObject *)&TileBaseType);

    //import_array();   /* required NumPy initialization */
}
