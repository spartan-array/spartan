#ifndef __EXTENT_H__
#define __EXTENT_H__
#include <stddef.h>
#include "cslice.h"
const int MAX_NDIM = 32;
class CExtent {
public:
    unsigned long long ul[MAX_NDIM];
    unsigned long long lr[MAX_NDIM];
    unsigned long long array_shape[MAX_NDIM];
    unsigned long long shape[MAX_NDIM];
    //unsigned long long *ul;
    //unsigned long long *lr;
    //unsigned long long *array_shape;
    //unsigned long long *shape;
    unsigned long long size;
    unsigned ndim;
    bool has_array_shape;

    CExtent(unsigned ndim, bool has_array_shape);
    ~CExtent();
    void init_info(void);
    Slice* to_slice(void); 
    unsigned long long ravelled_pos(void);
    unsigned to_global(unsigned long long idx, int axis);
    CExtent* add_dim(void);
    CExtent* clone(void);
};

CExtent* extent_create(unsigned long long ul[], 
                       unsigned long long lr[],
                       unsigned long long array_shape[],
                       unsigned ndim);

CExtent* extent_from_shape(unsigned long long shape[], unsigned ndim);

void unravelled_pos(unsigned long long idx, 
                    unsigned long long array_shape[], 
                    unsigned ndim, 
                    unsigned long long pos[]); // output

unsigned long long ravelled_pos(unsigned long long idx[],
                                unsigned long long array_shape[],
                                unsigned ndim);

bool all_nonzero_shape(unsigned long long shape[], unsigned ndim);

void find_rect(unsigned long long ravelled_ul,
               unsigned long long ravelled_lr,
               unsigned long long shape[],
               unsigned ndim,
               unsigned long long rect_ravelled_ul[], // output
               unsigned long long rect_ravelled_lr[]); // output

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
CExtent* compute_slice(CExtent* base, Slice idx[], unsigned idx_len);
CExtent* compute_slice_cy(CExtent* base, long long idx[], unsigned idx_len);

CExtent* offset_from(CExtent* base, CExtent* other);

void offset_slice(CExtent* base, CExtent* other, Slice slice[]);

CExtent* from_slice(Slice idx[], unsigned long long shape[], unsigned ndim);
CExtent* from_slice_cy(long long idx[], 
                       unsigned long long shape[], 
                       unsigned ndim);

void shape_for_reduction(unsigned long long input_shape[],
                         unsigned ndim, 
                         unsigned axis,
                         unsigned long long shape[]); // oputput

CExtent* index_for_reduction(CExtent *index, int axis);

bool shapes_match(CExtent *ex_a,  CExtent *ex_b);

CExtent* drop_axis(CExtent* ex, int axis);

void find_shape(CExtent **extents, int num_ex,
                unsigned long long shape[]); // output

bool is_complete(unsigned long long shape[], unsigned ndim, Slice slices[]);
#endif
