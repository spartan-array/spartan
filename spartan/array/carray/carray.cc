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
    data_source = new NpyMemManager(data);
}

CArray::CArray(npy_intp dimensions[], int nd, char type, char *data)
{
    init(dimensions, nd, type);
    this->data = data;
    data_source = NULL;
}

CArray::CArray(npy_intp dimensions[], int nd, char type, char *data, NpyMemManager *source)
{
    init(dimensions, nd, type);
    this->data = data;
    data_source = source;
}

CArray::CArray(CArray_RPC *rpc, NpyMemManager *source)
{
    init(rpc->dimensions, (int)rpc->nd, (char)rpc->item_type);
    data = rpc->data;
    data_source = source;
}

CArray::~CArray()
{
    if (data_source != NULL)
        delete data_source;
}

void
CArray::init(npy_intp dimensions[], int nd, char type)
{
    int i, type_size;

    type_size = npy_type_token_to_size(type);
    this->nd = nd;
    this->type = type;
    npy_intp stride;
    size = type_size;
    for (i = nd - 1; i >= 0; i++) {
        this->dimensions[i] = dimensions[i];
        if (i == nd - 1) {
            this->strides[i] = type_size;
        } else {
            this->strides[i] *=  this->strides[i + 1] * this->dimensions[i + 1];
        }
        size *= this->dimensions[i];
    }
}


/* dest = this[ex] */
void
CArray::copy(char *dest, CExtent *ex)
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
        memcpy(dest, data, size);
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

        npy_intp curr_idx[NPY_MAXDIMS], curr_pos, prev_pos = 0;
        for (i = 0; i < nd; i++) {
            curr_idx[i] = ex->ul[i];
        }
        all_size = ex->size * type_size; 

        char *source_data = this->data;
        do {
            curr_pos = ravelled_pos(curr_idx, ex->array_shape, nd);
            memcpy(dest, source_data + curr_pos, continous_size);

            for (i = last_sliced_dim; i >= 0; i++) {
                curr_idx[i] += 1;
                if (curr_idx[i] - ex->ul[i] < ex->shape[i]) {
                    break; 
                }
                curr_idx[i] = ex->ul[i];
            }
            prev_pos = curr_pos;
            dest += continous_size;
            all_size -= continous_size;
        } while(all_size > 0);
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
    npy_intp strides[NPY_MAXDIMS];
    NpyMemManager *manager = NULL;
    PyArrayObject *array;
    int type_num;

    type_num = npy_type_token_to_number(type);
    for (int i = nd - 1; i >= 0; i--) {
        if (i == nd - 1) {
            strides[i] = type_size;
        } else {
            strides[i] = strides[i + 1] * dimensions[i + 1];
        }
    }
    array = (PyArrayObject*)PyArray_New(NULL, nd, dimensions, type_num, strides, 
                                        data, 0, NPY_ARRAY_ALIGNED, NULL);
    assert(array != NULL);
    if (data_source != NULL) {
        manager = data_source;
    }
    PyObject *capsule = PyCapsule_New(manager, _CAPSULE_NAME, _capsule_destructor);
    assert(capsule);
    assert(PyArray_SetBaseObject(array, capsule) == 0);

    return (PyObject*)array;
}

void
CArray::to_carray_rpc(CArray_RPC *rpc, CExtent *ex)
{
    rpc->item_size = type_size;
    rpc->item_type = type;
    rpc->size = size;
    for (int i = 0; i < nd; ++i) {
        rpc->dimensions[i] = dimensions[i];    
    }
    rpc->nd = nd;
    copy(rpc->data, ex);
}
