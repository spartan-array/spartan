#ifndef __CTILE_H__
#define __CTILE_H__
#include <Python.h>
#include <numpy/arrayobject.h>
#include "../carray/cslice.h"
#include "../carray/carray.h"
#include "../carray/carray_reducer.h"

enum CTILE_TYPE {
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
    CArray_RPC array[0];
} CTile_RPC;

/**
 * CTile only initializes data(mask) when necessary. This can reduce 
 * transmission time when data is not required (such as creat() in 
 * distarray).
 */
class CTile {
public:

    CTile() : nd(0) {};
    CTile(npy_intp dimensions[], int nd, char dtype, CTILE_TYPE tile_type, CTILE_SPARSE_TYPE sparse_type);
    // This constructor will also set data pointers to rpc.
    // No need to call set_data again.
    CTile(CTile_RPC *rpc);
    ~CTile();

    void initialize(void);
    CExtent* slice_to_ex(CSliceIdx &idx);
    bool is_idx_complete(CSliceIdx &idx);

    bool set_data(CArray *dense, CArray *mask);
    bool set_data(CArray **sparse);
    void clear_data() {dense = mask = sparse[0] = sparse[1] = sparse[2] = NULL;};
    void update(CSliceIdx &idx, CTile &update_data, npy_intp reducer);

    // Backward compatible, just call to_tile_rpc.
    char* get(CSliceIdx &idx); 
    // This won't call Python APIs
    char* to_tile_rpc(CSliceIdx &idx);
    // This will call Python APIs
    PyObject* to_npy(void);

    // Setter and getter
    CTILE_TYPE get_type(void) { return type; };
    CTILE_SPARSE_TYPE get_sparse_type(void) { return sparse_type; };
    int get_nd(void) { return nd; };
    npy_intp* get_dimensions(void) { return dimensions; };
    char get_dtype(void) { return dtype; };
private:
    void reduce(CSliceIdx &idx, CTile &update, REDUCER reducer);

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

#endif
