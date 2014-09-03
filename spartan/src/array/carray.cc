#include <iostream>
#include <Python.h>
/* For Numpy C-API */
#define PY_ARRAY_UNIQUE_SYMBOL spartan_ctile_ARRAY_API
#include <numpy/arrayobject.h>
#include "carray.h"

int npy_type_size[] = {sizeof(npy_bool),
                       sizeof(npy_intp),
                       sizeof(npy_int),
                       sizeof(npy_uint),
                       sizeof(npy_long),
                       sizeof(npy_ulong),
                       sizeof(npy_longlong),
                       sizeof(npy_ulonglong),
                       sizeof(npy_float),
                       sizeof(npy_double)};

int npy_type_number[] = {NPY_BOOL, 
                         NPY_INTP,
                         NPY_INT, 
                         NPY_UINT, 
                         NPY_LONG,
                         NPY_ULONG,
                         NPY_LONGLONG,
                         NPY_ULONGLONG,
                         NPY_FLOAT,
                         NPY_DOUBLE};

char npy_type_token[] = {NPY_BOOLLTR, 
                         NPY_INTPLTR,
                         NPY_INTLTR, 
                         NPY_UINTLTR, 
                         NPY_LONGLTR,
                         NPY_ULONGLTR,
                         NPY_LONGLONGLTR,
                         NPY_ULONGLONGLTR,
                         NPY_FLOATLTR,
                         NPY_DOUBLELTR,
                         (char)-1};

std::map<char*, int> NpyMemManager::refcount;

CArray::CArray(npy_intp dimensions[], int nd, char type)
{
    init(dimensions, nd, type);
    data = (char*) malloc(size);
    Log_debug("CArray create a memory buffer %p", data);
    assert(this->data != NULL);
    memset(data, 0, size);
    data_source = new NpyMemManager(data, data, false, size);
}

CArray::CArray(npy_intp dimensions[], int nd, char type, char *buffer)
{
    init(dimensions, nd, type);
    data = buffer;
    Log_debug("CArray uses a memory buffer %p", data);
    data_source = new NpyMemManager(data, data, false, size);
}

CArray::CArray(npy_intp dimensions[], int nd, char type, char *data, PyArrayObject *source)
{
    init(dimensions, nd, type);
    this->data = data;
    Log_debug("CArray uses a memory buffer %p", data);
    data_source = new NpyMemManager((char*)source, data, true, size);
}

CArray::CArray(npy_intp dimensions[], int nd, char type, char *data, NpyMemManager *source)
{
    init(dimensions, nd, type);
    this->data = data;
    Log_debug("CArray uses a memory buffer %p", data);
    data_source = source;
}

CArray::CArray(CArray_RPC *rpc)
{
    std::cout << "CArray::" << __func__ << std::endl;
    init(rpc->dimensions, (int)rpc->nd, (char)rpc->item_type);
    if (rpc->is_npy_memmanager) { 
        data_source = (NpyMemManager*)(rpc->data);
        data = data_source->get_data();
        Log_debug("CArray uses a memory buffer %p", data);
    } else {
        data = rpc->data;
        Log_debug("CArray uses a memory buffer %p", data);
        data_source = new NpyMemManager(data, data, false, size);
    }
}

CArray::~CArray()
{
    if (data_source != NULL) {
        std::cout << __func__ << " data source = " << std::hex \
                  << (unsigned long)data_source << " data_source->source = " \
                  << (unsigned long)data_source->get_source() \
                  << " refcnt = " << data_source->get_refcount() << std::endl;
        delete data_source;
    }
}

void
CArray::init(npy_intp dimensions[], int nd, char type)
{
    int i;

    std::cout << "CArray::" << __func__ << (unsigned)type << std::endl;
    type_size = npy_type_token_to_size(type);
    this->nd = nd;
    this->type = type;
    size = type_size;
    for (i = nd - 1; i >= 0; i--) {
        this->dimensions[i] = dimensions[i];
        if (i == nd - 1) {
            strides[i] = type_size;
        } else {
            strides[i] = strides[i + 1] * this->dimensions[i + 1];
        }
        size *= this->dimensions[i];
    }
    std::cout << "CArray::" <<  __func__ << "end" << std::endl;
}


npy_intp
CArray::copy_slice(CExtent *ex, NpyMemManager **dest)
{
    bool full = true;
    int i;
    
    if (ex != NULL) {
        for (i = 0; i < nd; i++) {
            if (ex->ul[i] != 0 || ex->lr[i] != dimensions[i]) {
                full = false;
            }
        }
    }

    if (full) { /* Special case, no slice */
        *dest = new NpyMemManager(*data_source);
        Log_debug("Full copy. Do not have to do copy. *dest = %p", *dest);
        return size;
    } else {
        npy_intp copy_size, all_size;
        int i, last_sliced_dim;

        for (i = nd - 1; i >= 0; i--) {
            npy_intp dim; 
            
            dim = ex->lr[i] - ex->ul[i];
            dim = (dim == 0) ? 1 : dim;
            Log_debug("Copy info: ul[i] = %d, lr[i] = %d, dimensions[i] = %d",
                      ex->ul[i], ex->lr[i], dimensions[i]);
            if (dim != dimensions[i]) {
                break;
            }
        }

        last_sliced_dim = i;
        copy_size = strides[last_sliced_dim];
        Log_debug("Copy info : last_sliced_dim = %d, nd = %d", last_sliced_dim, nd);

        npy_intp curr_idx[NPY_MAXDIMS], curr_pos;
        for (i = 0; i < nd; i++) {
            curr_idx[i] = ex->ul[i];
        }
        all_size = ex->size * type_size; 
        npy_intp ret = all_size;

        char *source_data = data;
        char *buf = (char*)malloc(all_size);
        assert(buf != NULL);
        *dest = new NpyMemManager(buf, buf, false, all_size); 
        Log_debug("Not full copy. *dest = %p, buf = %p, all_size = %d, copy_size = %d",
                  *dest, buf, all_size, copy_size);
        do {
            curr_pos = ravelled_pos(curr_idx, ex->array_shape, nd);
            memcpy(buf, source_data + curr_pos, copy_size);

            for (i = last_sliced_dim; i >= 0; i--) {
                curr_idx[i] += 1;
                if (curr_idx[i] - ex->ul[i] < ex->shape[i]) {
                    break; 
                }
                curr_idx[i] = ex->ul[i];
            }
            buf += copy_size;
            all_size -= copy_size;
        } while(all_size > 0);
        if (all_size != 0) {
            Log_error("Something is wrong when doing copy. all_size = %d, copy_size = %d",
                      all_size, copy_size);
        }
        return ret;
    }
}

#define _CAPSULE_NAME "NpyMemManager"
static void _capsule_destructor(PyObject *o)
{
    NpyMemManager *manager = (NpyMemManager*)(PyCapsule_GetPointer(o, _CAPSULE_NAME));
    Log_debug("%s %p", __func__, manager->get_source());
    if (manager != NULL) {
        delete manager;
    }
}

PyObject*
CArray::to_npy(void)
{
    NpyMemManager *manager = NULL;
    PyArrayObject *array;
    int type_num;

    std::cout << "CArray::" << __func__ << std::endl;
    type_num = npy_type_token_to_number(type);
    std::cout << "Before PyArray_New " << type_num << " " << size << std::endl;
    array = (PyArrayObject*)PyArray_New(&PyArray_Type, nd, dimensions, type_num, strides, 
                                        data, size, NPY_CARRAY, NULL);
    std::cout << "PyArray_New" << std::endl;
    assert(array != NULL);
    if (data_source != NULL) {
        manager = new NpyMemManager(*data_source);
    }
    std::cout << "Before PyCapsule_New" << std::endl;
    PyObject *capsule = PyCapsule_New(manager, _CAPSULE_NAME, _capsule_destructor);
    std::cout << "PyCapsule_New" << std::endl;
    assert(capsule != NULL);
    if (PyArray_SetBaseObject(array, capsule) != 0)
        assert(false);

    return (PyObject*)array;
}

std::vector <char*>
CArray::to_carray_rpc(CExtent *ex)
{
    std::vector <char*> dest;
    std::cout << "CArray::" << __func__ << std::endl;
    CArray_RPC *rpc = (CArray_RPC*)malloc(sizeof(CArray_RPC));
    dest.push_back((char*)(new NpyMemManager((char*)rpc, (char*)rpc, 
                                             false, sizeof(CArray_RPC))));

    rpc->item_size = type_size;
    rpc->item_type = type;
    rpc->size = size;
    rpc->nd = nd;
    rpc->is_npy_memmanager = true;
    for (int i = 0; i < nd; ++i) {
        rpc->dimensions[i] = ex->lr[i] - ex->ul[i]; //dimensions[i];
        rpc->dimensions[i] = (rpc->dimensions[i] == 0) ? 1 : rpc->dimensions[i];
    }
    if (copy_slice(ex, (NpyMemManager**)(&(rpc->data))) != size) {
        assert(false);
    }
    dest.push_back(rpc->data);

    return dest;
}

// This is the entry when loading the shared library
static void _attach(void) __attribute__((constructor));
void _attach(void) {
    Py_Initialize();
    import_array();   /* required NumPy initialization */
}
