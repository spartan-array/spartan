#include <Python.h>
/* For Numpy C-API */
#define PY_ARRAY_UNIQUE_SYMBOL spartan_ctile_ARRAY_API
#define NO_IMPORT_ARRAY
#include <numpy/arrayobject.h>
#include <iostream>
#include <string>
#include <vector>
#include "ctile.h"
#include "carray_reducer.h"

CTile::CTile(npy_intp dimensions[], int nd, char dtype, 
             CTILE_TYPE tile_type, CTILE_SPARSE_TYPE sparse_type) 
    : py_c_refcount(0)
{
    int i;

    Log_debug("CTile::CTile(...)");
    initialized = false;
    type = (CTILE_TYPE)tile_type;
    this->sparse_type = (CTILE_SPARSE_TYPE)sparse_type;
    this->nd = nd;
    this->dtype = dtype;
    for (i = 0; i < nd; i++) {
        this->dimensions[i] = dimensions[i];
    }
    dense = NULL;
    dense_state = NULL;
    mask = NULL;
    mask_state = NULL;
    for (i = 0; i < 3; i++) {
        sparse[i] = NULL;
    }
}

CTile::CTile(CTile_RPC *rpc)
    : py_c_refcount(0)
{
    Log_debug("CTile::CTile(CTile_RPC)");
    type = (CTILE_TYPE)rpc->type;
    sparse_type = (CTILE_SPARSE_TYPE)rpc->sparse_type;
    initialized = rpc->initialized;
    nd = rpc->nd;
    for (int i = 0; i < rpc->nd; i++) {
        dimensions[i] = rpc->dimensions[i];
    }
    dtype = rpc->item_type;

    if (!initialized) {
        dense = NULL;
        dense_state = NULL;
        mask = NULL;
        mask_state = NULL;
        for (int i = 0; i < 3; i++) {
            sparse[i] = NULL;
        }
        return;
    }

    if (type != CTILE_SPARSE) {
        for (int i = 0; i < 3; i++) {
            sparse[i] = NULL;
        }
        dense = new CArray(rpc->array[0]);
        if (type == CTILE_MASKED) {
            mask = new CArray(rpc->array[1]);
        } else {
            mask = NULL;
        }
        dense_state = mask_state = NULL;
    } else {
        dense = dense_state = mask = mask_state = NULL;
        for (int i = 0; i < 3; i++) {
            sparse[i] = new CArray(rpc->array[i]);
        }
    }

}

CTile::~CTile()
{
    if (dense != NULL) delete dense;
    if (dense_state != NULL) delete dense_state;
    if (mask != NULL) delete mask;
    if (mask_state != NULL) delete mask_state;
    for (int i = 0; i < 3; i++) {
       if (sparse[i] != NULL) delete sparse[i]; 
    }
}

bool 
CTile::set_data(CArray *dense, CArray *mask)
{
    if (initialized || type == CTILE_MASKED) {
       return false; 
    }
    this->dense = dense;
    this->mask = mask;
    initialized = true;
    return true;
}

bool
CTile::set_data(CArray **sparse)
{
    if (initialized || type != CTILE_SPARSE) {
       return false; 
    }
    for (int i = 0; i < 3; i++) {
        this->sparse[i] = sparse[i];
    }
    initialized = true;
    return true;
}

void
CTile::initialize(void)
{
    Log_debug("CTile::initialize");
    if (type == CTILE_SPARSE) {
         sparse[0] = new CArray(dimensions, 1, NPY_INTPLTR);
         sparse[1] = new CArray(dimensions, 1, NPY_INTPLTR);
         sparse[2] = new CArray(dimensions, 1, dtype);
    } else {
        dense = new CArray(dimensions, nd, dtype);
        if (type == CTILE_MASKED) {
            mask = new CArray(dimensions, nd, dtype);
        }
    }
    initialized = true;
}

CExtent*
CTile::slice_to_ex(const CSliceIdx &idx) 
{
    return from_slice(idx, dimensions, nd);
}

bool
CTile::is_idx_complete(const CSliceIdx &idx)
{
    return is_complete(dimensions, nd, idx);
}

void 
CTile::reduce(const CSliceIdx &idx, CTile &update, REDUCER reducer)
{
    CExtent *ex = slice_to_ex(idx);
    bool full = is_idx_complete(idx);

    Log_debug("CTILE::reduce");
    if (nd == 0) { // Special case
        Log_debug("CTILE::reduce, scalar");
        scalar_outer_loop(dense, dense_state, update.dense, reducer);
    } else if (type == CTILE_DENSE || type == CTILE_MASKED) { 
        Log_debug("CTILE::reduce, dense or masked");
        if ((update.type == CTILE_DENSE && update.type == CTILE_MASKED) ||
             update.type == CTILE_DENSE) { 
            if (full) {
                Log_debug("CTILE::reduce, dense full");
                full_dense_outer_loop(dense, dense_state, update.dense, reducer);
            } else {
                Log_debug("CTILE::reduce, dense slice");
                slice_dense_outer_loop(dense, dense_state, update.dense, ex, reducer);
            }
        } else if (update.type != CTILE_SPARSE) {
            if (full) {
                full_dense_outer_loop(dense, dense_state, update.dense, reducer);
                full_dense_outer_loop(mask, mask_state, update.mask, REDUCER_OR);
            } else {
                slice_dense_outer_loop(dense, dense_state, update.dense, ex, reducer);
                slice_dense_outer_loop(mask, mask_state, update.mask, ex, REDUCER_OR);
            }
        } else { // SPARSE
            Log_debug("CTILE::reduce, dense_sparse");
            sparse_dense_outer_loop(dense, dense_state, update.sparse, ex, reducer);
        }

    } else if (type == CTILE_SPARSE) {
        Log_debug("CTILE::reduce, sparse");
        if (update.type == CTILE_DENSE || update.type == CTILE_MASKED) {
            assert(0);
        } else { 
            assert(0);
        }
    }
}

void 
CTile::update(const CSliceIdx &idx, CTile &update_data, npy_intp reducer)
{
    Log_debug("CTile::update, reducer = %X", (unsigned)reducer);
    if (!initialized) {
        initialize(); 
    }
    if (type != CTILE_SPARSE && dense_state == NULL) {
        dense_state = new CArray(dimensions, nd, NPY_BOOLLTR);
    } 
    if (type == CTILE_MASKED && mask_state == NULL) {
        mask_state = new CArray(dimensions, nd, NPY_BOOLLTR);
    }

    if (reducer >= REDUCER_BEGIN && reducer <= REDUCER_END) {
        reduce(idx, update_data, (REDUCER)reducer);
    } else {
        PyObject *old, *update, *subslice, *reducer_npy;

        PyObject *mod, *object;
        /* TODO: Get GIL here */
        mod = PyImport_ImportModule("spartan.array.tile");
        assert(mod != NULL);
        object = PyObject_GetAttrString(mod, "_internal_update");
        assert(object != NULL);

        old = to_npy();
        update = to_npy();
        subslice = PyTuple_New(nd);
        assert(subslice != NULL);
        for (int i = 0; i < nd; i++) {
            PyTuple_SET_ITEM(subslice, i, 
                             PySlice_New(PyLong_FromLongLong(idx.get_slice(i).start),
                                         PyLong_FromLongLong(idx.get_slice(i).stop),
                                         NULL));
        }
        reducer_npy = (PyObject*) reducer;
        /* TODO: Do we have to update our dense ? */
        PyObject_CallFunctionObjArgs(object, old, subslice, 
                                     update, reducer_npy, NULL);
    }
}

std::vector <char*>
CTile::get(const CSliceIdx &idx)
{
    return to_tile_rpc(idx);
}

std::vector <char*>
CTile::to_tile_rpc(const CSliceIdx &idx)
{
    Log_debug("CTile::%s, type = %d", __func__, type);
    std::vector<char*> dest;
    CExtent *ex = slice_to_ex(idx);
    dest.push_back((char*)(new bool(true)));
    CTile_RPC *rpc = new CTile_RPC;

    rpc->type = type;
    rpc->sparse_type = sparse_type;
    rpc->initialized = initialized;
    rpc->nd = nd;
    rpc->item_type = dtype;
    for (int i = 0; i < nd; i++) {
        rpc->dimensions[i] = dimensions[i];
    }
    dest.push_back((char*)(new NpyMemManager((char*)rpc, (char*)rpc, false, sizeof(CTile_RPC))));

    if (initialized && type != CTILE_SPARSE) {
        std::vector<char*> v = dense->to_carray_rpc(ex);
        dest.insert(dest.end(), v.begin(), v.end());
        rpc->count = 1;
        if (type == CTILE_MASKED) {
            Log_debug("Fetching a masked array");
            std::vector<char*> v = mask->to_carray_rpc(ex);
            dest.insert(dest.end(), v.begin(), v.end());
            rpc->count = 2;
        }
        Log_debug("The size of the vector of dense->to_carray_rpc is %u", v.size());

    } else if (initialized && type == CTILE_SPARSE) {
        if (is_idx_complete(idx)) {
            rpc->count = 3;
            for (int i = 0; i < 3; i++) {
                std::vector<char*> v = sparse[i]->to_carray_rpc();
                dest.insert(dest.end(), v.begin(), v.end());
            }
        } else {
            /* TODO: Sparse Slicing ..... */
            assert(0);
            //memcpy(base, &rpc, sizeof(CTile_RPC));
            return dest;
        }
    }
    delete ex;
    return dest;
}

PyObject*
CTile::to_npy(void)
{
    Log_debug("CTile::%s, type = %d", __func__, type);
    if (!initialized) {
        Log_debug("Getting a uninitialized tile");
        PyObject *mod, *object, *npy_dimensions;
        npy_dimensions = PyTuple_New(nd);
        assert(npy_dimensions != NULL);
        for (int i = 0; i < nd; i++) {
            PyTuple_SET_ITEM(npy_dimensions, i, PyLong_FromLongLong(dimensions[i]));  
        }
        if (type != CTILE_SPARSE) {
            mod = PyImport_ImportModule("numpy");
            assert(mod != NULL);
            object = PyObject_GetAttrString(mod, "ndarray");
        } else {
            mod = PyImport_ImportModule("scipy.sparse");
            assert(mod != NULL);
            if (sparse_type == CTILE_SPARSE_COO) {
                object = PyObject_GetAttrString(mod, "coo_matrix");
            } else if (sparse_type == CTILE_SPARSE_CSR) {
                object = PyObject_GetAttrString(mod, "csr_matrix");
            } else if (sparse_type == CTILE_SPARSE_CSC) {
                object = PyObject_GetAttrString(mod, "csc_matrix");
            } else {
                object = NULL;
            }
            assert(object != NULL);
        }
        PyObject *ret = PyObject_CallFunctionObjArgs(object, npy_dimensions, NULL);
        return ret;
    } else if (type == CTILE_DENSE) {
        return dense->to_npy();
    } else if (type == CTILE_MASKED) {
        PyObject *npy_dense = dense->to_npy();
        PyObject *npy_mask = mask->to_npy();
        PyObject *mod, *object, *sargs, *kword, *kwds, *ret;

        sargs = PyTuple_New(1);
        assert(sargs != NULL);
        PyTuple_SET_ITEM(sargs, 0, npy_dense);
        kword = PyString_FromString("mask");
        assert(kword != NULL);
        kwds = PyDict_New();
        assert(kwds != NULL);
        PyDict_SetItem(kwds, kword, npy_mask);
        mod = PyImport_ImportModule("numpy.ma");
        assert(mod != NULL);
        object = PyObject_GetAttrString(mod, "array");
        assert(object != NULL);
        ret = PyObject_Call(object, sargs, kwds);
        assert(ret != NULL);
        return ret;
    } else {
        PyObject *npy_sparse[3];
        PyObject *sp, *object, *tuple[3], *sargs, *kargs, *kwords, *ret;

        for (int i = 0; i < 3; i++) {
            npy_sparse[i] = sparse[i]->to_npy(); 
        }

        sp = PyImport_ImportModule("scipy.sparse");
        assert(sp != NULL);
        switch (sparse_type) {
        case CTILE_SPARSE_COO:
            for (int i = 0; i < 3; i++) {
                tuple[i] = PyTuple_New(2);
                assert(tuple[i] != NULL);
            }

            // coo_matrix((data, ij), shape)
            PyTuple_SET_ITEM(tuple[0], 0, PyLong_FromLongLong(dimensions[0]));
            PyTuple_SET_ITEM(tuple[0], 1, PyLong_FromLongLong(dimensions[1]));
            PyTuple_SET_ITEM(tuple[1], 0, npy_sparse[0]); // row
            PyTuple_SET_ITEM(tuple[1], 1, npy_sparse[1]); // col
            PyTuple_SET_ITEM(tuple[2], 0, npy_sparse[2]); // data
            PyTuple_SET_ITEM(tuple[2], 1, tuple[1]);
            sargs = PyTuple_New(1);
            assert(sargs != NULL);
            PyTuple_SET_ITEM(sargs, 0, tuple[2]);
            kwords = PyString_FromString("shape");
            assert(kwords != NULL);
            kargs = PyDict_New();
            assert(kargs != NULL);
            PyDict_SetItem(kargs, kwords, tuple[0]);
            std::cout << "ctile->to_npy() coo" << std::endl;
            object = PyObject_GetAttrString(sp, "coo_matrix");
            break;
        case CTILE_SPARSE_CSR:
        case CTILE_SPARSE_CSC:
            tuple[0] = PyTuple_New(2);
            assert(tuple[0] != NULL);
            tuple[1] = PyTuple_New(3);
            assert(tuple[1] != NULL);

            // csr_matrix/csc_matrix((data, indeices, indptr), shape)
            PyTuple_SET_ITEM(tuple[0], 0, PyLong_FromLongLong(dimensions[0]));
            PyTuple_SET_ITEM(tuple[0], 1, PyLong_FromLongLong(dimensions[1]));
            PyTuple_SET_ITEM(tuple[1], 1, npy_sparse[0]); // indices
            PyTuple_SET_ITEM(tuple[1], 2, npy_sparse[1]); // indptr
            PyTuple_SET_ITEM(tuple[1], 0, npy_sparse[2]); // data
            sargs = PyTuple_New(1);
            assert(sargs != NULL);
            PyTuple_SET_ITEM(sargs, 0, tuple[1]);
            kwords = PyString_FromString("shape");
            assert(kwords != NULL);
            kargs = PyDict_New();
            assert(kargs != NULL);
            PyDict_SetItem(kargs, kwords, tuple[0]);
            if (sparse_type == CTILE_SPARSE_CSR) {
                std::cout << "ctile->to_npy() csr" << std::endl;
                object = PyObject_GetAttrString(sp, "csr_matrix");
            } else {
                std::cout << "ctile->to_npy() csc" << std::endl;
                object = PyObject_GetAttrString(sp, "csc_matrix");
            }
            break;
        default:
            assert(0);
        }

        assert(object != NULL);
        ret = PyObject_Call(object, sargs, kargs);
        assert(dense != NULL);

        return ret;
    }
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
    return 0;
}

CTile*
ctile_creator(PyObject *args)
{
    PyObject *shape, *data;
    const char* dtype;
    unsigned long tile_type, sparse_type;

    if (!PyArg_ParseTuple(args, "OskkO", &shape, &dtype, &tile_type, &sparse_type, &data))
        return NULL;

    Log_debug("CTile::%s, type = %u", __func__, tile_type);
    int nd = PyTuple_Size(shape);
    npy_intp dimensions[NPY_MAXDIMS];
    for (int i = 0; i < nd; i++) {
        dimensions[i] = (npy_intp)get_longlong(PyTuple_GetItem(shape, i));
    }

    CTile *tile = new CTile(dimensions, nd, dtype[0], (CTILE_TYPE)tile_type, 
                            (CTILE_SPARSE_TYPE)sparse_type);
    if (tile == NULL)
        return NULL;

    if (data != Py_None) {
        assert(PyTuple_Check(data) != 0);
        if (tile_type != CTILE_SPARSE) {
            /* TODO: release dense */
            PyArrayObject *dense = (PyArrayObject*)PyTuple_GetItem(data, 0);
            dense = PyArray_GETCONTIGUOUS(dense);
            //PyArrayObject *target = dense;
            //while (target->base != NULL) {
                //// The memory is contiguous, but it is a viewed array
                //// such as np.arange(10).reshape(2, 5).
                //// TODO: Not sure if this covers everything we will encounter.
                //assert(PyArray_Check(target->base));
                //target = (PyArrayObject*) target->base;
            //}
            //char *base_data = target->data;
            //std::cout << "ctile_creater creates from an array " << target
                      //<< ", data is " << (unsigned long) base_data 
                      //<< ", dense data is " << (unsigned long) dense->data 
                      //<< std::endl;
            //for (int i = 0; i < 4; i++) {
                //std::cout << "OnlyForDebug " << i << " "
                          //<< *(unsigned long*)(base_data + i * 8) << std::endl;
            //}

            CArray *dense_array = new CArray(dense->dimensions, dense->nd,
                                             dense->descr->type, dense->data, dense);
            /* TODO: Should do the same transformation as dense */
            CArray *mask_array = NULL;
            if (tile_type == CTILE_MASKED) {
                PyArrayObject *mask = (PyArrayObject*)PyTuple_GetItem(data, 1);
                mask_array = new CArray(mask->dimensions, mask->nd,
                                        mask->descr->type, mask->data,
                                        mask);
            }
            tile->set_data(dense_array, mask_array);
        } else {
            CArray *sparse_array[3];
            /* TODO: Should do the same transformation as dense */
            for (int i = 0; i < 3; i++) {
                PyArrayObject *sparse = (PyArrayObject*)PyTuple_GetItem(data, i);
                sparse_array[i] = new CArray(sparse->dimensions, sparse->nd,
                                             sparse->descr->type, sparse->data,
                                             sparse);
            }
            tile->set_data(sparse_array);
        }
    }

    return tile;
}
