#include <Python.h>
#include <structmember.h>
#include <iostream>
#include "ctile.h"

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

typedef struct {
    PyObject_HEAD
    CTile *c_tile;
} Tile;

static bool 
is_integer(PyObject *o) {
    if (PyLong_Check(o) || PyInt_Check(o)) {
        return true;
    }
    return false;
}

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
}

static void
Tile_dealloc(PyObject *o)
{
    Tile *self = (Tile*) o; 
    std::cout << __func__ << self->c_tile << std::endl;
    delete self->c_tile;
    self->ob_type->tp_free((PyObject*)self);
}

//static PyObject *
//Tile_new(PyTypeObject *type, PyObject *args, PyObject *kwds)
//{
    //Tile *self;

    //self = (Tile*)type->tp_alloc(type, 0);
    //if (self != NULL) {
        ////self->c_tile = new CTile();
        //self->c_tile = NULL;
    //}

    //return (PyObject *) self;
//}

static PyObject*
Tile_reduce(PyObject* o)
{
    Tile *self = (Tile*) o;
    PyObject *mod, *result, *tuple, *obj, *ul, *lr, *array_shape;
    PyObject *shape, *dtype, *tile_type, *sparse_type;

    std::cout << "__reduce__" << std::endl;
    result = PyTuple_New(2);
    tuple = PyTuple_New(4);
    char dtype_str[] = {self->c_tile->get_dtype(), '\0'};
    dtype = PyString_FromString(dtype_str);
    tile_type = PyInt_FromLong((long)self->c_tile->get_type());
    sparse_type = PyInt_FromLong((long)self->c_tile->get_sparse_type());
    shape = PyTuple_New(self->c_tile->get_nd());
    std::cout << shape << " " << self->c_tile->get_nd() << std::endl;
    for (int i = 0; i < self->c_tile->get_nd(); i++) {
        PyTuple_SetItem(shape, i, PyLong_FromLongLong(self->c_tile->get_dimensions()[i]));
    }

    std::cout << "__reduce__" << std::endl;
    mod = PyImport_ImportModule("tile");
    obj = PyObject_GetAttrString(mod, "from_shape");
    Py_DECREF(mod);
    PyTuple_SetItem(tuple, 0, shape);
    PyTuple_SetItem(tuple, 1, dtype);
    PyTuple_SetItem(tuple, 2, tile_type);
    PyTuple_SetItem(tuple, 3,  sparse_type);
    PyTuple_SetItem(result, 0,  obj);
    PyTuple_SetItem(result, 1,  tuple);
    std::cout << "__reduce__" << std::endl;
    return result;
}

static PyObject *
Tile_get(PyObject* o, PyObject* args)
{
    Tile *self = (Tile*)o;
    PyObject *slice;
    CSlice c_slice[NPY_MAXDIMS];

    //std::cout << __func__ << std::endl;
    if (!PyArg_ParseTuple(args, "O",  &slice))
        return NULL;

    CSliceIdx cslice_idx(slice, self->c_tile->get_nd(), self->c_tile->get_dimensions());

    CTile_RPC *rpc = (CTile_RPC*) self->c_tile->get(cslice_idx);
    CTile *ctile = new CTile(rpc);
    PyObject *ret = ctile->to_npy();

    if (ret == NULL) {
            free(rpc);
    } else {
            return ret;
    }
}

static PyObject *
Tile__update(PyObject* o, PyObject* args)
{
    Tile *self = (Tile*)o;
    PyObject *slice, *data, *tile_type, *sparse_type, *reducer;
    CSlice cslice[NPY_MAXDIMS];

    //std::cout << __func__ << std::endl;
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
        PyArrayObject *dense = (PyArrayObject*)PyTuple_GetItem(data, 0);
        CTile tile(dense->dimensions, dense->nd, dense->descr->type, ttype, CTILE_SPARSE_NONE);
        CArray *dense_array = new CArray(dense->dimensions, dense->nd,
                                         dense->descr->type, dense->data);
        CArray *mask_array = NULL;
        if (ttype == CTILE_MASKED) {
            PyArrayObject *mask = (PyArrayObject*)PyTuple_GetItem(data, 1);
            mask_array = new CArray(mask->dimensions, mask->nd,
                                    mask->descr->type, mask->data);
        }
        tile.set_data(dense_array, mask_array);
        self->c_tile->update(cslice_idx, tile, npy_reducer);
    } else {
        CArray *sparse_array[3];
        for (int i = 0; i < 3; i++) {
            PyArrayObject *sparse = (PyArrayObject*)PyTuple_GetItem(data, i);
            sparse_array[i] = new CArray(sparse->dimensions, sparse->nd,
                                         sparse->descr->type, sparse->data);
        }
        CExtent *ex = from_slice(cslice_idx, self->c_tile->get_dimensions(),
                                 self->c_tile->get_nd());
        CTile tile(ex->shape, sparse_array[3]->get_nd(), sparse_array[3]->get_type(),
                   ttype, stype);
        tile.set_data(sparse_array);
        self->c_tile->update(cslice_idx, tile, npy_reducer);
    }

    return o;
}

static int 
Tile_init(PyObject *o, PyObject *args, PyObject *kwds)
{
    Tile *self = (Tile*)o;
    PyObject *shape, *dtype_obj, *tile_type, *sparse_type, *data, *test;

    std::cout << __func__ << std::endl;
    if (!PyArg_ParseTuple(args, "OOOOOO", &shape, &dtype_obj, &tile_type, &sparse_type, &data, &test))
        return -1;
    
    int nd = PyTuple_Size(shape);
    npy_intp dimensions[NPY_MAXDIMS];
    char *dtype;
    for (int i = 0; i < nd; i++) {
        dimensions[i] = (npy_intp)get_longlong(PyTuple_GetItem(shape, i));
    }
    std::cout << __func__ << "0" << std::endl;
    dtype = PyString_AsString(dtype_obj);
    CTILE_TYPE ttype = (CTILE_TYPE)get_longlong(tile_type);
    CTILE_SPARSE_TYPE stype = (CTILE_SPARSE_TYPE)get_longlong(sparse_type);

    std::cout << __func__ << "1" << std::endl;
    CTile *tile = new CTile(dimensions, nd, dtype[0], ttype, stype);
    std::cout << __func__ << "2" << std::endl;
    if (data != Py_None) {
        if (ttype != CTILE_SPARSE) {
            PyArrayObject *dense = (PyArrayObject*)PyTuple_GetItem(data, 0);
            CArray *dense_array = new CArray(dense->dimensions, dense->nd,
                                             dense->descr->type, dense->data);
            CArray *mask_array = NULL;
            if (ttype == CTILE_MASKED) {
                PyArrayObject *mask = (PyArrayObject*)PyTuple_GetItem(data, 1);
                mask_array = new CArray(mask->dimensions, mask->nd,
                                        mask->descr->type, mask->data);
            }
            tile->set_data(dense_array, mask_array);
        } else {
            CArray *sparse_array[3];
            for (int i = 0; i < 3; i++) {
                PyArrayObject *sparse = (PyArrayObject*)PyTuple_GetItem(data, i);
                sparse_array[i] = new CArray(sparse->dimensions, sparse->nd,
                                             sparse->descr->type, sparse->data);
            }
            tile->set_data(sparse_array);
        }
    }
    std::cout << __func__ << "4" << std::endl;
    self->c_tile = tile;
    return 0;
}

static PyMemberDef Tile_members[] = {
    {NULL} /* Sentinel */
};

static PyGetSetDef Tile_getseters[] = {
    {NULL} /* Sentinel */
};

static PyMethodDef Tile_methods[] = {
    {"__reduce__", (PyCFunction)Tile_reduce, METH_NOARGS,
     "__reduce__"},
    {(char*)"get", (PyCFunction)Tile_get, METH_VARARGS,
     (char*)"Get data from the tile."},
    {(char*)"_update", (PyCFunction)Tile__update, METH_VARARGS,
     (char*)"Update data to the tile."},
    {NULL} /* Sentinel */
};

static PyTypeObject TileType = {
    PyObject_HEAD_INIT(NULL)
    0,                           /* ob_size */
    "tile.Tile",                 /* tp_name */
    sizeof(Tile),                /* tp_basicsize */
    0,                           /* tp_itemsize */
    Tile_dealloc,                /* tp_dealloc */
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
    Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE,          /* tp_flags */
    "tile object",               /* tp_doc */
    0,                           /* tp_traverse */
    0,                           /* tp_clear */
    0,                           /* tp_richcompare */
    0,                           /* tp_weaklistoffset */
    0,                           /* tp_iter */
    0,                           /* tp_iternext */
    Tile_methods,                /* tp_methods */
    Tile_members,                /* tp_members */
    Tile_getseters,              /* tp_getset */
    0,                           /* tp_base */
    0,                           /* tp_dict */
    0,                           /* tp_descr_get */
    0,                           /* tp_descr_set */
    0,                           /* tp_dictoffset */
    Tile_init,                   /* tp_init */
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

    TileType.tp_new = PyType_GenericNew;
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
    TileType.tp_dict = tp_dict;
    if (PyType_Ready(&TileType) < 0)
        return;

    m = Py_InitModule3("_ctile_py_if", extent_methods,
                       "Python interface for ctile module");

    Py_INCREF(&TileType);
    PyModule_AddObject(m, "Tile", (PyObject *)&TileType);
}

