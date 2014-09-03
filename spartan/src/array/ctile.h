#ifndef __CTILE_H__
#define __CTILE_H__
#include <Python.h>
#include <numpy/arrayobject.h>
#include <string>
#include <vector>
#include "cslice.h"
#include "carray.h"
#include "carray_reducer.h"
#include "rpc/marshal.h"
#include "base/logging.h"

enum CTILE_TYPE {
    CTILE_EMPTY = 0,
    CTILE_DENSE = 1,
    CTILE_MASKED = 2,
    CTILE_SPARSE = 3,
};

enum CTILE_SPARSE_TYPE {
    CTILE_SPARSE_NONE = 0,
    CTILE_SPARSE_COO = 1,
    CTILE_SPARSE_CSC = 2,
    CTILE_SPARSE_CSR = 3,
};

#define NEXT_CARRAY_RPC(curr, curr_size, next) \
    next = (CArray_RPC*)((char*)(curr) + sizeof(CArray_RPC) + (curr_size));

typedef struct CTile_RPC_t {
    npy_int32 type;
    npy_int32 sparse_type;
    npy_int32 initialized;
    npy_int32 nd;
    npy_int64 dimensions[NPY_MAXDIMS];
    npy_int32 item_type; // This is a char, use 32bits for alignment.
    npy_int32 count;

    /**
     * The fields may repeat.
     * For a dense array, there is only one array.
     * For a masked array, there are two arrays.
     * For a sparse array, there are three arrays.
     * For a sparse array, the 'dimension' field means the dimension
     * of the whole sparse array. Hence, users need to caculate the
     * 'dimension' for the three matrices from item_size and size.
     */
     CArray_RPC* array[3];
} CTile_RPC;

inline CTile_RPC* vector_to_ctile_rpc(std::vector<char*> &buffers) {
    bool is_npy_memmanager= *(bool*)buffers[0];
    delete (bool*)buffers[0];

    CTile_RPC *rpc;
    if (is_npy_memmanager) {
        NpyMemManager* mgr;
        mgr = (NpyMemManager*)buffers[1];
        rpc = (CTile_RPC*)mgr->get_data();
        mgr->clear();
        delete mgr;
        rpc->array[0] = rpc->array[1] = rpc->array[2] = NULL;
        for (unsigned i = 2, array_idx = 0; i < buffers.size(); i += 2, array_idx++) {
            assert((i + 1) < buffers.size());
            mgr = (NpyMemManager*)buffers[i];
            rpc->array[array_idx] = (CArray_RPC*) mgr->get_data();
            mgr->clear();
            delete mgr;
            rpc->array[array_idx]->is_npy_memmanager = true;
            rpc->array[array_idx]->data = buffers[i + 1];
        }
    } else {
        rpc = (CTile_RPC*)buffers[1];
        rpc->array[0] = rpc->array[1] = rpc->array[2] = NULL;
        std::cout << __func__ << std::endl;
        for (unsigned i = 2, array_idx = 0; i < buffers.size(); i += 2, array_idx++) {
            assert((i + 1) < buffers.size());
            rpc->array[array_idx] = (CArray_RPC*) buffers[i];
            rpc->array[array_idx]->is_npy_memmanager = false;
            rpc->array[array_idx]->data = buffers[i + 1];
            std::cout <<  __func__ << " " << std::hex << (unsigned long)buffers[i + 1] << std::endl;
        }
    }

    return rpc;
}

inline void release_ctile_rpc(CTile_RPC* rpc)
{
    for (int i = 0; i < 3; ++i) {
        if (rpc->array[i] != NULL) {
            if (rpc->array[i]->is_npy_memmanager) {
                delete (NpyMemManager*)(rpc->array[i]);
            } else {
                delete rpc->array[i];
            }
        }
    }
    free(rpc);
}

/**
 * CTile only initializes data(mask) when necessary. This can reduce
 * transmission time when data is not required (such as creat() in
 * distarray).
 */
class CTile {
public:

    CTile()
        : py_c_refcount(0), nd(0), dense(NULL), mask(NULL), sparse {NULL, NULL, NULL} {};
    CTile(npy_intp dimensions[], int nd, char dtype,
          CTILE_TYPE tile_type, CTILE_SPARSE_TYPE sparse_type);
    // This constructor will also set data pointers to rpc.
    // No need to call set_data again.
    CTile(CTile_RPC *rpc);
    ~CTile();

    void initialize(void);

    bool set_data(CArray *dense, CArray *mask);
    bool set_data(CArray **sparse);
    void clear_data() {dense = mask = sparse[0] = sparse[1] = sparse[2] = NULL;};
    void update(const CSliceIdx &idx, CTile &update_data, npy_intp reducer);

    // Backward compatible, just call to_tile_rpc.
    std::vector <char*> get(const CSliceIdx &idx);
    // This won't call Python APIs
    std::vector <char*> to_tile_rpc(const CSliceIdx &idx);

    // This will call Python APIs
    PyObject* to_npy(void);

    // Setter and getter
    CTILE_TYPE get_type(void) { return type; }
    CTILE_SPARSE_TYPE get_sparse_type(void) { return sparse_type; }
    int get_nd(void) { return nd; }
    npy_intp* get_dimensions(void) { return dimensions; }
    char get_dtype(void) { return dtype; }

    void increase_py_c_refcount (void) { py_c_refcount++; }
    void decrease_py_c_refcount (void) { py_c_refcount--; }
    bool can_release (void) { return py_c_refcount == 0; }

    // For RPC marshal
    friend rpc::Marshal& operator<<(rpc::Marshal&, const CTile&);
    friend rpc::Marshal& operator>>(rpc::Marshal&m, CTile& o);
private:
    CExtent* slice_to_ex(const CSliceIdx &idx);
    bool is_idx_complete(const CSliceIdx &idx);
    void reduce(const CSliceIdx &idx, CTile &update, REDUCER reducer);

    /**
     * This reference count is used to determine whether this CTile is used
     * by Python only or by both C++ and Python.
     */
    int py_c_refcount;

    CTILE_TYPE type;
    CTILE_SPARSE_TYPE sparse_type;
    int nd;
    npy_intp dimensions[NPY_MAXDIMS];
    char dtype;
    bool initialized;

    CArray *dense;
    CArray *mask;
    CArray *sparse[3]; // COO, CSR, CSC all use three arrays to represent data.

};

inline rpc::Marshal& operator<<(rpc::Marshal& m, const CTile& o)
{
    m.write(&o.type, sizeof(o.type));
    m.write(&o.sparse_type, sizeof(o.sparse_type));
    m << o.nd;
    m.write(o.dimensions, sizeof(npy_intp) * NPY_MAXDIMS);
    m.write(&o.dtype, sizeof(o.dtype));
    m.write(&o.initialized, sizeof(o.initialized));
    Log_debug("Marshal::%s , type = %d, nd = %d, initialized = %u, dtype = %c",
              __func__, o.type, o.nd, o.initialized, o.dtype);
    if (o.initialized) {
        if (o.type == CTILE_DENSE) {
            m << *(o.dense);
        } else if (o.type == CTILE_MASKED) {
            m << *(o.dense);
            m << *(o.mask);
        } else if (o.type == CTILE_SPARSE) {
            for (int i = 0; i < 3; ++i) {
                m << *(o.sparse[i]);
            }
        }
    }
    return m;
}

inline rpc::Marshal& operator>>(rpc::Marshal&m, CTile& o)
{
    m.read(&o.type, sizeof(o.type));
    m.read(&o.sparse_type, sizeof(o.sparse_type));
    m >> o.nd;
    m.read(o.dimensions, sizeof(npy_intp) * NPY_MAXDIMS);
    m.read(&o.dtype, sizeof(o.dtype));
    m.read(&o.initialized, sizeof(o.initialized));
    Log_debug("Marshal::%s , type = %d, nd = %d, initialized = %u, dtype = %c",
              __func__, o.type, o.nd, o.initialized, o.dtype);
    if (o.initialized) {
        if (o.type == CTILE_DENSE) {
            o.dense = new CArray();
            m >> *(o.dense);
        } else if (o.type == CTILE_MASKED) {
            o.dense = new CArray();
            m >> *(o.dense);
            o.mask = new CArray();
            m >> *(o.mask);
        } else if (o.type == CTILE_SPARSE) {
            for (int i = 0; i < 3; ++i) {
                o.sparse[i] = new CArray();
                m >> *(o.sparse[i]);
            }
        }
    }

    return m;
}

CTile* ctile_creator(PyObject *args);
#endif
