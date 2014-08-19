#ifndef __EXTENT_H__
#define __EXTENT_H__
#include <stddef.h>
#include <Python.h>
#include <numpy/arrayobject.h>
#include <numpy/ndarrayobject.h>
#include <numpy/numpyconfig.h>
#include "cslice.h"

class CExtent {
public:
    npy_intp ul[NPY_MAXDIMS];
    npy_intp lr[NPY_MAXDIMS];
    npy_intp array_shape[NPY_MAXDIMS];
    npy_intp shape[NPY_MAXDIMS];
    npy_intp size;
    int ndim;
    bool has_array_shape;

    CExtent(int ndim, bool has_array_shape);
    ~CExtent();
    void init_info(void);
    CSliceIdx* to_slice(void); 
    npy_intp ravelled_pos(void);
    npy_intp to_global(npy_intp idx);
    CExtent* add_dim(void);
    CExtent* clone(void);
};

CExtent* extent_create(npy_intp ul[], 
                       npy_intp lr[],
                       npy_intp array_shape[],
                       int ndim);

CExtent* extent_from_shape(npy_intp shape[], int ndim);

void unravelled_pos(npy_intp idx, 
                    npy_intp array_shape[], 
                    int ndim, 
                    npy_intp pos[]); // output

npy_intp ravelled_pos(npy_intp idx[], npy_intp array_shape[], int ndim);

bool all_nonzero_shape(npy_intp shape[], int ndim);

void find_rect(npy_intp ravelled_ul,
               npy_intp ravelled_lr,
               npy_intp shape[],
               int ndim,
               npy_intp rect_ravelled_ul[], // output
               npy_intp rect_ravelled_lr[]); // output

CExtent* intersection(CExtent* a, CExtent* b);

/**
 * This is different from the find_overlapping() in Python.
 * C++ doesn't have yield! Use static is dangerous.
 */
CExtent* find_overlapping(CExtent* extent, CExtent* region);

/**
 * This API uses a two dimensions array to simulate a tuple of slices.
 * idx[i][0] is ith's start and idx[i][1] is ith's stop.
 */
CExtent* compute_slice(CExtent* base, CSliceIdx& idx);

CExtent* offset_from(CExtent* base, CExtent* other);

void offset_slice(CExtent* base, CExtent* other, CSlice slice[]);

CExtent* from_slice(CSliceIdx &idx, npy_intp shape[], int ndim);

void shape_for_reduction(npy_intp input_shape[],
                         int ndim, 
                         int axis,
                         npy_intp shape[]); // oputput

CExtent* index_for_reduction(CExtent *index, int axis);

bool shapes_match(CExtent *ex_a,  CExtent *ex_b);

CExtent* drop_axis(CExtent* ex, int axis);

void find_shape(CExtent **extents, int num_ex,
                npy_intp shape[]); // output

bool is_complete(npy_intp shape[], int ndim, CSliceIdx &idx);
#endif
