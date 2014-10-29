#ifndef __CARRAY_REDUCER_H__
#define __CARRAY_REDUCER_H__
#include "carray.h"

enum REDUCER {
    REDUCER_BEGIN   = 0xF000,
    REDUCER_REPLACE = 0xF000,
    REDUCER_ADD     = 0xF001,
    REDUCER_MUL     = 0xF002,
    REDUCER_MAXIMUM = 0xF003,
    REDUCER_MINIMUM = 0xF004,
    REDUCER_AND     = 0xF005,
    REDUCER_OR      = 0xF006,
    REDUCER_XOR     = 0xF007,
    REDUCER_END     = 0xF0FF,
}; 


void scalar_outer_loop(CArray *ip1, CArray *ip1_state, CArray *ip2, REDUCER reducer);
void slice_dense_outer_loop(CArray *ip1, CArray *ip1_state, CArray *ip2, 
                            CExtent *ex, REDUCER reducer);
void trivial_dense_outer_loop(CArray *ip1, CArray *ip1_state, CArray *ip2, REDUCER reducer);
void sparse_dense_outer_loop(CArray *dense, CArray *dense_state, CArray *sparse[3], 
                             CExtent *ex, REDUCER reducer);

#endif
