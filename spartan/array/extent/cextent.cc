#include <cassert>
#include "cextent.h"

#include <iostream>
CExtent::CExtent(unsigned ndim, bool has_array_shape) 
{
    this->ndim = ndim;
    this->has_array_shape = has_array_shape;
}

CExtent::~CExtent()
{
}

void CExtent::init_info(void)
{
    size = 1;
    for (unsigned i = 0; i < ndim; i++) {
        shape[i] = lr[i] - ul[i];
        if (shape[i] == 0) {
            shape[i] = 1; 
        }
        size *= shape[i];
    }      
}

Slice* CExtent::to_slice(void) 
{
   Slice* slices; 

   for (unsigned i = 0; i < ndim; i++) {
        slices[i].start = ul[i];
        slices[i].stop = lr[i];
        slices[i].step = 1;
   }

   return slices;
}

unsigned long long CExtent::ravelled_pos(void)
{
    return ::ravelled_pos(ul, array_shape, ndim);
}

unsigned CExtent::to_global(unsigned long long idx, int axis)
{
    unsigned long long local_idx[MAX_NDIM];
    if (axis >= 0) {
       return idx + ul[axis];
    }

    unravelled_pos(idx, shape, ndim, local_idx);
    for (unsigned i = 0; i < ndim; i++) {
        local_idx[i] += ul[i]; 
    }
    return ::ravelled_pos(local_idx, array_shape, ndim);
};

CExtent* CExtent::add_dim(void) 
{
    unsigned long long ul[MAX_NDIM], lr[MAX_NDIM], array_shape[MAX_NDIM];

    for (unsigned i = 0; i < ndim; i++) {
        ul[i] = this->ul[i];
        lr[i] = this->lr[i];
        array_shape[i] = this->array_shape[i];
    }
    ul[ndim] = 0;
    lr[ndim] = 1;
    array_shape[ndim] = 1;

    return extent_create(ul, lr, array_shape, ndim + 1);
};

CExtent* CExtent::clone(void) {
    return extent_create(ul, lr, array_shape, ndim);
};

CExtent* extent_create(unsigned long long ul[], 
                      unsigned long long lr[],
                      unsigned long long array_shape[],
                      unsigned ndim)
{
    CExtent *ex = new CExtent(ndim, (array_shape != NULL));    

    ex->size = 1;
    for (unsigned i = 0; i < ndim; i++) {
        if (ul[i] >= lr[i]) {
           delete ex;
           return NULL;
        }

        ex->ul[i] = ul[i];
        ex->lr[i] = lr[i];
        ex->shape[i] = lr[i] - ul[i];
        if (ex->shape[i] == 0) {
            ex->shape[i] = 1; 
        }
        ex->size *= ex->shape[i];
        if (array_shape != NULL) {
            ex->array_shape[i] = array_shape[i]; 
        }
    }
    return ex;
}

CExtent* extent_from_shape(unsigned long long shape[], unsigned ndim)
{
    unsigned long long ul[MAX_NDIM], lr[MAX_NDIM];

    for (unsigned i = 0; i < ndim; i++) {
       ul[i] = 0;
       lr[i] = shape[i];
    }

    return extent_create(ul, lr, shape, ndim);
}

void unravelled_pos(unsigned long long idx, 
                    unsigned long long array_shape[], 
                    unsigned ndim, 
                    unsigned long long pos[]) // output
{
    for (int i = ndim - 1; i >= 0; i--) {
        pos[i] = idx % array_shape[i];
        idx /= ndim;
    }
}

unsigned long long ravelled_pos(unsigned long long idx[],
                                unsigned long long array_shape[],
                                unsigned ndim)
{
    unsigned rpos = 0;
    unsigned mul = 1;

    for (int i = ndim - 1; i >= 0; i--) {
        rpos += mul * idx[i];
        mul *= array_shape[i];
    }

    return rpos;
}

bool all_nonzero_shape(unsigned long long shape[],
                       unsigned ndim)
{
    for (unsigned i = 0; i < ndim; i++) {
        if (shape[i] == 0)
            return false;
    }
    return true;
}

void find_rect(unsigned long long ravelled_ul,
               unsigned long long ravelled_lr,
               unsigned long long shape[],
               unsigned ndim,
               unsigned long long rect_ravelled_ul[], // output
               unsigned long long rect_ravelled_lr[]) // output
{
    if (shape[ndim - 1] == 1 || 
        ravelled_ul / shape[ndim - 1] == ravelled_lr / shape[ndim - 1]) {
        *rect_ravelled_ul = ravelled_ul;
        *rect_ravelled_lr = ravelled_lr;
    } else {
        int div = 1;
        for (unsigned i = 1; i < ndim; i++) {
            div *= i; 
        }
        *rect_ravelled_ul = ravelled_ul - (ravelled_ul % div);
        *rect_ravelled_lr = ravelled_lr - (div - ravelled_lr % div) % div -1;
    }
}

CExtent* intersection(CExtent* a, CExtent* b)
{
    unsigned long long ul[MAX_NDIM];
    unsigned long long lr[MAX_NDIM];

    if (a == NULL || b == NULL) {
       return NULL; 
    }
    for (unsigned i = 0; i < a->ndim ; i++) {
        if ((a->has_array_shape xor a->has_array_shape) ||
            (a->has_array_shape && a->array_shape[i] != b->array_shape[i])) {
            assert(0);
        }
        if (b->lr[i] < a->ul[i]) return NULL;
        if (a->lr[i] < b->ul[i]) return NULL;
        ul[i] = (a->ul[i] >= b->ul[i]) ? a->ul[i] : b->ul[i];
        lr[i] = (a->lr[i] < b->lr[i]) ? a->lr[i] : b->lr[i];
    }
    return extent_create(ul, lr, a->array_shape, a->ndim);
}

/**
 * This is different from the find_overlapping() in Python.
 * C++ doesn't have yield! Use static is dangerous.
 */
CExtent* find_overlapping(CExtent* extent, CExtent* region)
{
    return intersection(extent, region);
}

CExtent* compute_slice(CExtent* base, Slice idx[], unsigned idx_len)
{
    unsigned long long ul[MAX_NDIM];
    unsigned long long lr[MAX_NDIM];

    for (unsigned i = 0; i < base->ndim; i++) {
        if (i >= idx_len) {
            ul[i] = base->ul[i];
            lr[i] = base->lr[i];
        } else {
            unsigned long long dim = base->shape[i];
            if (idx[i].start < 0) idx[i].start += dim;
            if (idx[i].stop < 0) idx[i].stop += dim;
            ul[i] = base->ul[i] + idx[i].start;
            lr[i] = base->ul[i] + idx[i].stop;
        }
    }
    return extent_create(ul, lr, base->array_shape, base->ndim);
}

CExtent* compute_slice_cy(CExtent* base, long long idx[], unsigned idx_len)
{
    Slice slices[MAX_NDIM];

    for (unsigned i = 0; i < idx_len; i++) {
        slices[i].start = idx[i * 2]; 
        slices[i].stop = idx[i * 2 + 1]; 
    }
    return compute_slice(base, slices, idx_len);
}

CExtent* offset_from(CExtent* base, CExtent* other)
{
    unsigned long long ul[MAX_NDIM];
    unsigned long long lr[MAX_NDIM];

    for (unsigned i = 0; i < base->ndim; i++) {
        if (other->ul[i] < base->ul[i] || other->lr[i] > base->lr[i]) {
            return NULL;
        }
        ul[i] = other->ul[i] - base->ul[i];
        lr[i] = other->lr[i] - base->ul[i];
    }
    return extent_create(ul, lr, other->array_shape, base->ndim);
}

void offset_slice(CExtent* base, CExtent* other, Slice slice[])
{
    Slice* slices;

    slice = new Slice[base->ndim];
    for (unsigned i = 0; i < base->ndim; i++) {
        slice[i].start = other->ul[i] - base->ul[i];
        slice[i].stop = other->lr[i] - base->lr[i];
        slice[i].step = 1;
    }
}

CExtent* from_slice(Slice idx[], unsigned long long shape[], unsigned ndim)
{
    unsigned long long ul[MAX_NDIM], lr[MAX_NDIM];

    for (unsigned i = 0; i < ndim; i++) {
        unsigned long long dim = shape[i];
        if (idx[i].start >= dim) assert(0);
        if (idx[i].stop >= dim) assert(0);
        if (idx[i].start < 0) idx[i].start += dim;
        if (idx[i].stop < 0) idx[i].stop += dim;
        ul[i] = idx[i].start;
        lr[i] = idx[i].stop;
    }

    return extent_create(ul, lr, shape, ndim);
}

CExtent* from_slice_cy(long long idx[], 
                       unsigned long long shape[], 
                       unsigned ndim)
{
    Slice slices[MAX_NDIM];

    for (unsigned i = 0; i < ndim; i++) {
        slices[i].start = idx[i * 2]; 
        slices[i].stop = idx[i * 2 + 1]; 
    }
    return from_slice(slices, shape, ndim);
}


void shape_for_reduction(unsigned long long input_shape[],
                         unsigned ndim, 
                         unsigned axis,
                         unsigned long long shape[]) // oputput
{
    unsigned i;
    for (i = 0; i < axis; i++) {
        shape[i] = input_shape[i];
    }

    for (i = axis + 1; i < ndim; i++) {
        shape[i - 1] =  input_shape[i];
    }
}

CExtent* index_for_reduction(CExtent *index, int axis)
{
    return drop_axis(index, axis);
}

bool shapes_match(CExtent *ex_a,  CExtent *ex_b)
{
    if (ex_a->ndim != ex_b->ndim) {
        return false;
    }

    for (unsigned i = 0; i < ex_a->ndim; i++) {
        if (ex_a->shape[i] != ex_b->shape[i]) {
           return false; 
        }
    }

    return true;
}

CExtent* drop_axis(CExtent* ex, int axis)
{
    unsigned long long ul[MAX_NDIM], lr[MAX_NDIM], shape[MAX_NDIM];
    int i;

    if (axis < 0) {
        axis = ex->ndim + axis;
    }

    for (i = 0; i < axis; i++) {
        shape[i] = ex->array_shape[i]; 
        ul[i] = ex->ul[i];
        lr[i] = ex->lr[i];
    }

    for (i = axis + 1; i < ex->ndim; i++) {
        shape[i - 1] = ex->array_shape[i]; 
        ul[i - 1] = ex->ul[i];
        lr[i - 1] = ex->lr[i];
    }

    return extent_create(ul, lr, shape, ex->ndim - 1);
}

void find_shape(CExtent **extents, int num_ex,
                unsigned long long *shape) // output
{
    int i, j;

    for (i = 0; i < extents[0]->ndim; i++) {
       shape[i] = 1; 
    }

    for (i = 0; i < num_ex; i++) {
        CExtent *ex = extents[i];
        for (j = 0; j < ex->ndim; j++) {
            if (shape[i] < ex->lr[i]) {
                shape[i] = ex->lr[i];
            }
        }
    }
}

bool is_complete(unsigned long long shape[], unsigned ndim, Slice slices[])
{
    for (unsigned i = 0; i < ndim; i++) {
        if (slices[i].start != 0) return false;
        if (slices[i].stop < shape[i]) return false;
    }

    return true;
}

