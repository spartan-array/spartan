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
    assert(this->data != NULL);
    memset(data, 0, size);
    data_source = new NpyMemManager(data, data, false, size);
}

CArray::CArray(npy_intp dimensions[], int nd, char type, char *buffer)
{
    init(dimensions, nd, type);
    data = buffer;
    data_source = new NpyMemManager(data, data, false, size);
}

CArray::CArray(npy_intp dimensions[], int nd, char type, char *data, PyArrayObject *source)
{
    init(dimensions, nd, type);
    this->data = data;
    data_source = new NpyMemManager((char*)source, data, true, size);
}

CArray::CArray(npy_intp dimensions[], int nd, char type, char *data, NpyMemManager *source)
{
    init(dimensions, nd, type);
    this->data = data;
    data_source = source;
}

CArray::CArray(CArray_RPC *rpc)
{
    std::cout << __func__ << " " << rpc << std::endl;
    std::cout << __func__ << " is_npy_memmanager = " << rpc->is_npy_memmanager << std::endl;
    std::cout << __func__ << " nd = " << (int)rpc->nd << std::endl;
    std::cout << __func__ << " size = " << rpc->size << std::endl;

    for (int i = 0; i < (int)rpc->nd; ++i) {
        std::cout << __func__ << " " << rpc->dimensions[i] << std::endl;  
    }
    init(rpc->dimensions, (int)rpc->nd, (char)rpc->item_type);
    if (rpc->is_npy_memmanager) { 
        data_source = (NpyMemManager*)(rpc->data);
        data = data_source->get_data();
        data_source = new NpyMemManager(data, data, false, size);
    } else {
        data = rpc->data;
        data_source = new NpyMemManager(data, data, false, size);
    }
}

CArray::~CArray()
{
    std::cout << __func__ << std::endl;
    if (data_source != NULL) {
        std::cout << __func__ << " " << (unsigned long) data_source->get_source() << std::endl;
        delete data_source;
    }
}

void
CArray::init(npy_intp dimensions[], int nd, char type)
{
    int i;

    std::cout << __func__ << (unsigned)type << std::endl;
    type_size = npy_type_token_to_size(type);
    std::cout << "type_size " << type_size << std::endl;
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
    std::cout << __func__ << "end" << std::endl;
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
    std::cout << __func__ << " " << full << " " << (void*)dest << " " << (void*)data << std::endl;

    if (full) { /* Special case, no slice */
        *dest = new NpyMemManager(*data_source);
        return size;
    } else {
        npy_intp continous_size, all_size;
        int i, last_sliced_dim;

        for (i = nd - 1; i >= 0; i--) {
            npy_intp dim; 
            
            dim = ex->lr[i] - ex->ul[i];
            dim = (dim == 0) ? 1 : dim;
            if (dim != dimensions[i]) {
                break;
            }
        }

        last_sliced_dim = i;
        if (last_sliced_dim == nd - 1) {
            continous_size = dimensions[nd - 1];
        } else {
            continous_size = 1;
            for (i = last_sliced_dim ; i < nd; i++) {
                continous_size *= dimensions[i];
            }
        }

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
        do {
            curr_pos = ravelled_pos(curr_idx, ex->array_shape, nd);
            memcpy(buf, source_data + curr_pos, continous_size);

            for (i = last_sliced_dim; i >= 0; i--) {
                curr_idx[i] += 1;
                if (curr_idx[i] - ex->ul[i] < ex->shape[i]) {
                    break; 
                }
                curr_idx[i] = ex->ul[i];
            }
            buf += continous_size;
            all_size -= continous_size;
        } while(all_size > 0);
        return ret;
    }
}

#define _CAPSULE_NAME "NpyMemManager"
static void _capsule_destructor(PyObject *o)
{
    NpyMemManager *manager = (NpyMemManager*)(PyCapsule_GetPointer(o, _CAPSULE_NAME));
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

    std::cout << "CArray::" << __func__ << " " << nd << " " << type_size << std::endl;
    type_num = npy_type_token_to_number(type);
    std::cout << "Before PyArray_New " << type_num << " " << size << " " << (void*)&PyArray_Type << std::endl;
    array = (PyArrayObject*)PyArray_New(&PyArray_Type, nd, dimensions, type_num, strides, 
                                        data, size, NPY_CARRAY, NULL);
    std::cout << "PyArray_New" << std::endl;
    assert(array != NULL);
    if (data_source != NULL) {
        manager = data_source;
    }
    std::cout << "Before PyCapsule_New" << std::endl;
    PyObject *capsule = PyCapsule_New(manager, _CAPSULE_NAME, _capsule_destructor);
    std::cout << "PyCapsule_New" << std::endl;
    assert(capsule);
    if (PyArray_SetBaseObject(array, capsule) != 0)
        assert(false);

    return (PyObject*)array;
}

std::vector <char*>
CArray::to_carray_rpc(CExtent *ex)
{
    std::vector <char*> dest;
    std::cout << __func__ << type_size << " " << type << " " << nd <<std::endl;
    CArray_RPC *rpc = (CArray_RPC*)malloc(sizeof(CArray_RPC));
    dest.push_back((char*)(new NpyMemManager((char*)rpc, (char*)rpc, false, sizeof(CArray_RPC))));

    rpc->item_size = type_size;
    rpc->item_type = type;
    rpc->size = size;
    rpc->nd = nd;
    rpc->is_npy_memmanager = true;
    for (int i = 0; i < nd; ++i) {
        rpc->dimensions[i] = dimensions[i];    
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
