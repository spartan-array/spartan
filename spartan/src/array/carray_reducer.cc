/**
 * This file implements the most important reduce functions for Spartan.
 * The implementation mimic Numpy's umath to loop over the input matrices.
 * However, carray_reducer is a special case and doesn't have to be general.
 * Restrictions:
 * 1. In Spartan, tile reducer only accept two parameters, the original matrix
 *    and updated matrix, and write output back to the original matrix.
 * 2. In Spartan, the original and updated matrix must be continous. And the
 *    idx (slices) only apply to the original matrix.
 * 3. "idx" (slices) must form a rectangle.
 * 4. The types of the original matrix and updated matrix must be the same.
 *    Spartan only does tile reducer when each worker produce data to the same
 *    tile. Consider that each worker execute the same mapper, the data must
 *    be the same type.
 * 5. Only supports two-dimensions sparse arrays (just like scipy.sparse does now).
 */

#include <iostream>
#define PY_ARRAY_UNIQUE_SYMBOL spartan_ctile_ARRAY_API
#define NO_IMPORT_ARRAY
#include "carray.h"
#include "carray_reducer.h"
#include "cextent.h"

typedef npy_int _RP_TYPE_;
const char _RP_TYPELTR_ = NPY_INTLTR;

/**
 * Scalar reducer
 */

void
BOOL_scalar_replace(char **args)
{
    char *ip1 = args[0], *ip2 = args[1];

    //npy_bool in1 = *((npy_bool*)ip1);
    npy_bool in2 = *((npy_bool*)ip2);
    *((npy_bool*)ip1) = in2;
}

void
BOOL_scalar_add(char **args)
{
    char *ip1 = args[0], *ip2 = args[1];

    npy_bool in1 = *((npy_bool*)ip1);
    npy_bool in2 = *((npy_bool*)ip2);
    *((npy_bool*)ip1) = in1 + in2;
}

void
BOOL_scalar_multiply(char **args)
{
    char *ip1 = args[0], *ip2 = args[1];

    npy_bool in1 = *((npy_bool*)ip1);
    npy_bool in2 = *((npy_bool*)ip2);
    *((npy_bool*)ip1) = in1 * in2;
}

void
BOOL_scalar_maximum(char **args)
{
    char *ip1 = args[0], *ip2 = args[1];

    npy_bool in1 = *((npy_bool*)ip1);
    npy_bool in2 = *((npy_bool*)ip2);
    *((npy_bool*)ip1) = (in1 >= in2) ? in1 : in2;
}

void
BOOL_scalar_minimum(char **args)
{
    char *ip1 = args[0], *ip2 = args[1];

    npy_bool in1 = *((npy_bool*)ip1);
    npy_bool in2 = *((npy_bool*)ip2);
    *((npy_bool*)ip1) = (in1 < in2) ? in1 : in2;
}

/*
void
BOOL_scalar_and(char **args)
{
    char *ip1 = args[0], *ip2 = args[1];

    npy_bool in1 = *((npy_bool*)ip1);
    npy_bool in2 = *((npy_bool*)ip2);
    *((npy_bool*)ip1) = in1 & in2;
}

void
BOOL_scalar_or(char **args)
{
    char *ip1 = args[0], *ip2 = args[1];

    npy_bool in1 = *((npy_bool*)ip1);
    npy_bool in2 = *((npy_bool*)ip2);
    *((npy_bool*)ip1) = in1 | in2;
}

void
BOOL_scalar_xor(char **args)
{
    char *ip1 = args[0], *ip2 = args[1];

    npy_bool in1 = *((npy_bool*)ip1);
    npy_bool in2 = *((npy_bool*)ip2);
    *((npy_bool*)ip1) = in1 xor in2;
}
*/

void
INT_scalar_replace(char **args)
{
    char *ip1 = args[0], *ip2 = args[1];

    //npy_int in1 = *((npy_int*)ip1);
    npy_int in2 = *((npy_int*)ip2);
    *((npy_int*)ip1) = in2;
}

void
INT_scalar_add(char **args)
{
    char *ip1 = args[0], *ip2 = args[1];

    npy_int in1 = *((npy_int*)ip1);
    npy_int in2 = *((npy_int*)ip2);
    *((npy_int*)ip1) = in1 + in2;
}

void
INT_scalar_multiply(char **args)
{
    char *ip1 = args[0], *ip2 = args[1];

    npy_int in1 = *((npy_int*)ip1);
    npy_int in2 = *((npy_int*)ip2);
    *((npy_int*)ip1) = in1 * in2;
}

void
INT_scalar_maximum(char **args)
{
    char *ip1 = args[0], *ip2 = args[1];

    npy_int in1 = *((npy_int*)ip1);
    npy_int in2 = *((npy_int*)ip2);
    *((npy_int*)ip1) = (in1 >= in2) ? in1 : in2;
}

void
INT_scalar_minimum(char **args)
{
    char *ip1 = args[0], *ip2 = args[1];

    npy_int in1 = *((npy_int*)ip1);
    npy_int in2 = *((npy_int*)ip2);
    *((npy_int*)ip1) = (in1 < in2) ? in1 : in2;
}

/*
void
INT_scalar_and(char **args)
{
    char *ip1 = args[0], *ip2 = args[1];

    npy_int in1 = *((npy_int*)ip1);
    npy_int in2 = *((npy_int*)ip2);
    *((npy_int*)ip1) = in1 & in2;
}

void
INT_scalar_or(char **args)
{
    char *ip1 = args[0], *ip2 = args[1];

    npy_int in1 = *((npy_int*)ip1);
    npy_int in2 = *((npy_int*)ip2);
    *((npy_int*)ip1) = in1 | in2;
}

void
INT_scalar_xor(char **args)
{
    char *ip1 = args[0], *ip2 = args[1];

    npy_int in1 = *((npy_int*)ip1);
    npy_int in2 = *((npy_int*)ip2);
    *((npy_int*)ip1) = in1 xor in2;
}
*/

void
UINT_scalar_replace(char **args)
{
    char *ip1 = args[0], *ip2 = args[1];

    //npy_uint in1 = *((npy_uint*)ip1);
    npy_uint in2 = *((npy_uint*)ip2);
    *((npy_uint*)ip1) = in2;
}

void
UINT_scalar_add(char **args)
{
    char *ip1 = args[0], *ip2 = args[1];

    npy_uint in1 = *((npy_uint*)ip1);
    npy_uint in2 = *((npy_uint*)ip2);
    *((npy_uint*)ip1) = in1 + in2;
}

void
UINT_scalar_multiply(char **args)
{
    char *ip1 = args[0], *ip2 = args[1];

    npy_uint in1 = *((npy_uint*)ip1);
    npy_uint in2 = *((npy_uint*)ip2);
    *((npy_uint*)ip1) = in1 * in2;
}

void
UINT_scalar_maximum(char **args)
{
    char *ip1 = args[0], *ip2 = args[1];

    npy_uint in1 = *((npy_uint*)ip1);
    npy_uint in2 = *((npy_uint*)ip2);
    *((npy_uint*)ip1) = (in1 >= in2) ? in1 : in2;
}

void
UINT_scalar_minimum(char **args)
{
    char *ip1 = args[0], *ip2 = args[1];

    npy_uint in1 = *((npy_uint*)ip1);
    npy_uint in2 = *((npy_uint*)ip2);
    *((npy_uint*)ip1) = (in1 < in2) ? in1 : in2;
}

/*
void
UINT_scalar_and(char **args)
{
    char *ip1 = args[0], *ip2 = args[1];

    npy_uint in1 = *((npy_uint*)ip1);
    npy_uint in2 = *((npy_uint*)ip2);
    *((npy_uint*)ip1) = in1 & in2;
}

void
UINT_scalar_or(char **args)
{
    char *ip1 = args[0], *ip2 = args[1];

    npy_uint in1 = *((npy_uint*)ip1);
    npy_uint in2 = *((npy_uint*)ip2);
    *((npy_uint*)ip1) = in1 | in2;
}

void
UINT_scalar_xor(char **args)
{
    char *ip1 = args[0], *ip2 = args[1];

    npy_uint in1 = *((npy_uint*)ip1);
    npy_uint in2 = *((npy_uint*)ip2);
    *((npy_uint*)ip1) = in1 xor in2;
}
*/

void
LONGLONG_scalar_replace(char **args)
{
    char *ip1 = args[0], *ip2 = args[1];

    //npy_longlong in1 = *((npy_longlong*)ip1);
    npy_longlong in2 = *((npy_longlong*)ip2);
    *((npy_longlong*)ip1) = in2;
}

void
LONGLONG_scalar_add(char **args)
{
    char *ip1 = args[0], *ip2 = args[1];

    npy_longlong in1 = *((npy_longlong*)ip1);
    npy_longlong in2 = *((npy_longlong*)ip2);
    *((npy_longlong*)ip1) = in1 + in2;
}

void
LONGLONG_scalar_multiply(char **args)
{
    char *ip1 = args[0], *ip2 = args[1];

    npy_longlong in1 = *((npy_longlong*)ip1);
    npy_longlong in2 = *((npy_longlong*)ip2);
    *((npy_longlong*)ip1) = in1 * in2;
}

void
LONGLONG_scalar_maximum(char **args)
{
    char *ip1 = args[0], *ip2 = args[1];

    npy_longlong in1 = *((npy_longlong*)ip1);
    npy_longlong in2 = *((npy_longlong*)ip2);
    *((npy_longlong*)ip1) = (in1 >= in2) ? in1 : in2;
}

void
LONGLONG_scalar_minimum(char **args)
{
    char *ip1 = args[0], *ip2 = args[1];

    npy_longlong in1 = *((npy_longlong*)ip1);
    npy_longlong in2 = *((npy_longlong*)ip2);
    *((npy_longlong*)ip1) = (in1 < in2) ? in1 : in2;
}

/*
void
LONGLONG_scalar_and(char **args)
{
    char *ip1 = args[0], *ip2 = args[1];

    npy_longlong in1 = *((npy_longlong*)ip1);
    npy_longlong in2 = *((npy_longlong*)ip2);
    *((npy_longlong*)ip1) = in1 & in2;
}

void
LONGLONG_scalar_or(char **args)
{
    char *ip1 = args[0], *ip2 = args[1];

    npy_longlong in1 = *((npy_longlong*)ip1);
    npy_longlong in2 = *((npy_longlong*)ip2);
    *((npy_longlong*)ip1) = in1 | in2;
}

void
LONGLONG_scalar_xor(char **args)
{
    char *ip1 = args[0], *ip2 = args[1];

    npy_longlong in1 = *((npy_longlong*)ip1);
    npy_longlong in2 = *((npy_longlong*)ip2);
    *((npy_longlong*)ip1) = in1 xor in2;
}
*/

void
ULONGLONG_scalar_replace(char **args)
{
    char *ip1 = args[0], *ip2 = args[1];

    //npy_ulonglong in1 = *((npy_ulonglong*)ip1);
    npy_ulonglong in2 = *((npy_ulonglong*)ip2);
    *((npy_ulonglong*)ip1) = in2;
}

void
ULONGLONG_scalar_add(char **args)
{
    char *ip1 = args[0], *ip2 = args[1];

    npy_ulonglong in1 = *((npy_ulonglong*)ip1);
    npy_ulonglong in2 = *((npy_ulonglong*)ip2);
    *((npy_ulonglong*)ip1) = in1 + in2;
}

void
ULONGLONG_scalar_multiply(char **args)
{
    char *ip1 = args[0], *ip2 = args[1];

    npy_ulonglong in1 = *((npy_ulonglong*)ip1);
    npy_ulonglong in2 = *((npy_ulonglong*)ip2);
    *((npy_ulonglong*)ip1) = in1 * in2;
}

void
ULONGLONG_scalar_maximum(char **args)
{
    char *ip1 = args[0], *ip2 = args[1];

    npy_ulonglong in1 = *((npy_ulonglong*)ip1);
    npy_ulonglong in2 = *((npy_ulonglong*)ip2);
    *((npy_ulonglong*)ip1) = (in1 >= in2) ? in1 : in2;
}

void
ULONGLONG_scalar_minimum(char **args)
{
    char *ip1 = args[0], *ip2 = args[1];

    npy_ulonglong in1 = *((npy_ulonglong*)ip1);
    npy_ulonglong in2 = *((npy_ulonglong*)ip2);
    *((npy_ulonglong*)ip1) = (in1 < in2) ? in1 : in2;
}

/*
void
ULONGLONG_scalar_and(char **args)
{
    char *ip1 = args[0], *ip2 = args[1];

    npy_ulonglong in1 = *((npy_ulonglong*)ip1);
    npy_ulonglong in2 = *((npy_ulonglong*)ip2);
    *((npy_ulonglong*)ip1) = in1 & in2;
}

void
ULONGLONG_scalar_or(char **args)
{
    char *ip1 = args[0], *ip2 = args[1];

    npy_ulonglong in1 = *((npy_ulonglong*)ip1);
    npy_ulonglong in2 = *((npy_ulonglong*)ip2);
    *((npy_ulonglong*)ip1) = in1 | in2;
}

void
ULONGLONG_scalar_xor(char **args)
{
    char *ip1 = args[0], *ip2 = args[1];

    npy_ulonglong in1 = *((npy_ulonglong*)ip1);
    npy_ulonglong in2 = *((npy_ulonglong*)ip2);
    *((npy_ulonglong*)ip1) = in1 xor in2;
}
*/

void
FLOAT_scalar_replace(char **args)
{
    char *ip1 = args[0], *ip2 = args[1];

    //npy_float in1 = *((npy_float*)ip1);
    npy_float in2 = *((npy_float*)ip2);
    *((npy_float*)ip1) = in2;
}

void
FLOAT_scalar_add(char **args)
{
    char *ip1 = args[0], *ip2 = args[1];

    npy_float in1 = *((npy_float*)ip1);
    npy_float in2 = *((npy_float*)ip2);
    *((npy_float*)ip1) = in1 + in2;
}

void
FLOAT_scalar_multiply(char **args)
{
    char *ip1 = args[0], *ip2 = args[1];

    npy_float in1 = *((npy_float*)ip1);
    npy_float in2 = *((npy_float*)ip2);
    *((npy_float*)ip1) = in1 * in2;
}

void
FLOAT_scalar_maximum(char **args)
{
    char *ip1 = args[0], *ip2 = args[1];

    npy_float in1 = *((npy_float*)ip1);
    npy_float in2 = *((npy_float*)ip2);
    *((npy_float*)ip1) = (in1 >= in2) ? in1 : in2;
}

void
FLOAT_scalar_minimum(char **args)
{
    char *ip1 = args[0], *ip2 = args[1];

    npy_float in1 = *((npy_float*)ip1);
    npy_float in2 = *((npy_float*)ip2);
    *((npy_float*)ip1) = (in1 < in2) ? in1 : in2;
}

/*
void
FLOAT_scalar_and(char **args)
{
    char *ip1 = args[0], *ip2 = args[1];

    npy_float in1 = *((npy_float*)ip1);
    npy_float in2 = *((npy_float*)ip2);
    *((npy_float*)ip1) = in1 & in2;
}

void
FLOAT_scalar_or(char **args)
{
    char *ip1 = args[0], *ip2 = args[1];

    npy_float in1 = *((npy_float*)ip1);
    npy_float in2 = *((npy_float*)ip2);
    *((npy_float*)ip1) = in1 | in2;
}

void
FLOAT_scalar_xor(char **args)
{
    char *ip1 = args[0], *ip2 = args[1];

    npy_float in1 = *((npy_float*)ip1);
    npy_float in2 = *((npy_float*)ip2);
    *((npy_float*)ip1) = in1 xor in2;
}
*/

void
DOUBLE_scalar_replace(char **args)
{
    char *ip1 = args[0], *ip2 = args[1];

    //npy_double in1 = *((npy_double*)ip1);
    npy_double in2 = *((npy_double*)ip2);
    *((npy_double*)ip1) = in2;
}

void
DOUBLE_scalar_add(char **args)
{
    char *ip1 = args[0], *ip2 = args[1];

    npy_double in1 = *((npy_double*)ip1);
    npy_double in2 = *((npy_double*)ip2);
    *((npy_double*)ip1) = in1 + in2;
}

void
DOUBLE_scalar_multiply(char **args)
{
    char *ip1 = args[0], *ip2 = args[1];

    npy_double in1 = *((npy_double*)ip1);
    npy_double in2 = *((npy_double*)ip2);
    *((npy_double*)ip1) = in1 * in2;
}

void
DOUBLE_scalar_maximum(char **args)
{
    char *ip1 = args[0], *ip2 = args[1];

    npy_double in1 = *((npy_double*)ip1);
    npy_double in2 = *((npy_double*)ip2);
    *((npy_double*)ip1) = (in1 >= in2) ? in1 : in2;
}

void
DOUBLE_scalar_minimum(char **args)
{
    char *ip1 = args[0], *ip2 = args[1];

    npy_double in1 = *((npy_double*)ip1);
    npy_double in2 = *((npy_double*)ip2);
    *((npy_double*)ip1) = (in1 < in2) ? in1 : in2;
}

/*
void
DOUBLE_scalar_and(char **args)
{
    char *ip1 = args[0], *ip2 = args[1];

    npy_double in1 = *((npy_double*)ip1);
    npy_double in2 = *((npy_double*)ip2);
    *((npy_double*)ip1) = in1 & in2;
}

void
DOUBLE_scalar_or(char **args)
{
    char *ip1 = args[0], *ip2 = args[1];

    npy_double in1 = *((npy_double*)ip1);
    npy_double in2 = *((npy_double*)ip2);
    *((npy_double*)ip1) = in1 | in2;
}

void
DOUBLE_scalar_xor(char **args)
{
    char *ip1 = args[0], *ip2 = args[1];

    npy_double in1 = *((npy_double*)ip1);
    npy_double in2 = *((npy_double*)ip2);
    *((npy_double*)ip1) = in1 xor in2;
}
*/


typedef void (*scalar_reducer)(char**);
scalar_reducer scalar_replace_functions[] = {BOOL_scalar_replace, INT_scalar_replace, UINT_scalar_replace, LONGLONG_scalar_replace, ULONGLONG_scalar_replace, FLOAT_scalar_replace, DOUBLE_scalar_replace, NULL};
scalar_reducer scalar_add_functions[] = {BOOL_scalar_add, INT_scalar_add, UINT_scalar_add, LONGLONG_scalar_add, ULONGLONG_scalar_add, FLOAT_scalar_add, DOUBLE_scalar_add, NULL};
scalar_reducer scalar_multiply_functions[] = {BOOL_scalar_multiply, INT_scalar_multiply, UINT_scalar_multiply, LONGLONG_scalar_multiply, ULONGLONG_scalar_multiply, FLOAT_scalar_multiply, DOUBLE_scalar_multiply, NULL};
scalar_reducer scalar_maximum_functions[] = {BOOL_scalar_maximum, INT_scalar_maximum, UINT_scalar_maximum, LONGLONG_scalar_maximum, ULONGLONG_scalar_maximum, FLOAT_scalar_maximum, DOUBLE_scalar_maximum, NULL};
scalar_reducer scalar_minimum_functions[] = {BOOL_scalar_minimum, INT_scalar_minimum, UINT_scalar_minimum, LONGLONG_scalar_minimum, ULONGLONG_scalar_minimum, FLOAT_scalar_minimum, DOUBLE_scalar_minimum, NULL};
//scalar_reducer scalar_and_functions[] = {BOOL_scalar_and, INT_scalar_and, UINT_scalar_and, LONGLONG_scalar_and, ULONGLONG_scalar_and, FLOAT_scalar_and, DOUBLE_scalar_and, NULL};
//scalar_reducer scalar_or_functions[] = {BOOL_scalar_or, INT_scalar_or, UINT_scalar_or, LONGLONG_scalar_or, ULONGLONG_scalar_or, FLOAT_scalar_or, DOUBLE_scalar_or, NULL};
//scalar_reducer scalar_xor_functions[] = {BOOL_scalar_xor, INT_scalar_xor, UINT_scalar_xor, LONGLONG_scalar_xor, ULONGLONG_scalar_xor, FLOAT_scalar_xor, DOUBLE_scalar_xor, NULL};

char scalar_funcs_type[] = {NPY_BOOLLTR, NPY_INTLTR, NPY_UINTLTR, NPY_LONGLONGLTR, NPY_ULONGLONGLTR, NPY_FLOATLTR, NPY_DOUBLELTR, ' '};
/* This must be sync with REDUCER enumeration */
scalar_reducer* scalar_functions[] = {
                                     scalar_replace_functions,  /* REDUCER_REPLACE */
                                     scalar_add_functions,      /* REDUCER_ADD */
                                     scalar_multiply_functions, /* REDUCER_MUL */
                                     scalar_maximum_functions,  /* REDUCER_MAXIMUM */
                                     scalar_minimum_functions,  /* REDUCER_MINIMUM */
                                     /*scalar_and_functions,      [> REDUCER_AND <]*/
                                     /*scalar_or_functions,       [> REDUCER_OR <]*/
                                     /*scalar_xor_functions,      [> REDUCER_XOR <]*/
                                     NULL,
                                   };

scalar_reducer
select_scalar_reducer(REDUCER reducer, char type)
{
    int i;

    for (i = 0; scalar_funcs_type[i] != ' '; i++) {
        if (scalar_funcs_type[i] == type)
            return scalar_functions[reducer - REDUCER_BEGIN][i];
    }
    return NULL;
}

void
scalar_outer_loop(CArray *ip1, CArray *ip2, REDUCER reducer)
{
    std::cout << __func__ << std::endl;
    char *arrays[2] = {ip1->get_data(), ip2->get_data()};

    scalar_reducer func = select_scalar_reducer(reducer, ip1->get_type());
    func(arrays);
}

/**
 * Dense array to dense array
 */
#define BINARY_DENSE_LOOP\
    char *ip1 = args[0], *ip2 = args[1];\
    npy_intp sp = steps[0];\
    npy_intp n = dimensions[0];\
    npy_intp i;\
    for(i = 0; i < n; i++, ip1 += sp, ip2 += sp)



void
BOOL_dense_replace(char **args, npy_intp *dimensions, npy_intp *steps)
{
    BINARY_DENSE_LOOP {
        npy_bool in2 = *((npy_bool*)ip2);
        *((npy_bool*)ip1) = in2;
    }
}

void
BOOL_dense_add(char **args, npy_intp *dimensions, npy_intp *steps)
{
    BINARY_DENSE_LOOP {
        npy_bool in1 = *((npy_bool*)ip1);
        npy_bool in2 = *((npy_bool*)ip2);
        *((npy_bool*)ip1) = in1 + in2;
    }
}

void
BOOL_dense_multiply(char **args, npy_intp *dimensions, npy_intp *steps)
{
    BINARY_DENSE_LOOP {
        npy_bool in1 = *((npy_bool*)ip1);
        npy_bool in2 = *((npy_bool*)ip2);
        *((npy_bool*)ip1) = in1 * in2;
    }
}

void
BOOL_dense_maximum(char **args, npy_intp *dimensions, npy_intp *steps)
{
    BINARY_DENSE_LOOP {
        npy_bool in1 = *((npy_bool*)ip1);
        npy_bool in2 = *((npy_bool*)ip2);
        *((npy_bool*)ip1) = (in1 >= in2) ? in1 : in2;
    }
}

void
BOOL_dense_minimum(char **args, npy_intp *dimensions, npy_intp *steps)
{
    BINARY_DENSE_LOOP {
        npy_bool in1 = *((npy_bool*)ip1);
        npy_bool in2 = *((npy_bool*)ip2);
        *((npy_bool*)ip1) = (in1 <= in2) ? in1 : in2;
    }
}

/*
void
BOOL_dense_and(char **args, npy_intp *dimensions, npy_intp *steps)
{
    BINARY_DENSE_LOOP {
        npy_bool in1 = *((npy_bool*)ip1);
        npy_bool in2 = *((npy_bool*)ip2);
        *((npy_bool*)ip1) = in1 and in2;
    }
}

void
BOOL_dense_or(char **args, npy_intp *dimensions, npy_intp *steps)
{
    BINARY_DENSE_LOOP {
        npy_bool in1 = *((npy_bool*)ip1);
        npy_bool in2 = *((npy_bool*)ip2);
        *((npy_bool*)ip1) = in1 or in2;
    }
}

void
BOOL_dense_xor(char **args, npy_intp *dimensions, npy_intp *steps)
{
    BINARY_DENSE_LOOP {
        npy_bool in1 = *((npy_bool*)ip1);
        npy_bool in2 = *((npy_bool*)ip2);
        *((npy_bool*)ip1) = in1 xor in2;
    }
}
*/

void
INT_dense_replace(char **args, npy_intp *dimensions, npy_intp *steps)
{
    BINARY_DENSE_LOOP {
        npy_int in2 = *((npy_int*)ip2);
        *((npy_int*)ip1) = in2;
    }
}

void
INT_dense_add(char **args, npy_intp *dimensions, npy_intp *steps)
{
    BINARY_DENSE_LOOP {
        npy_int in1 = *((npy_int*)ip1);
        npy_int in2 = *((npy_int*)ip2);
        *((npy_int*)ip1) = in1 + in2;
    }
}

void
INT_dense_multiply(char **args, npy_intp *dimensions, npy_intp *steps)
{
    BINARY_DENSE_LOOP {
        npy_int in1 = *((npy_int*)ip1);
        npy_int in2 = *((npy_int*)ip2);
        *((npy_int*)ip1) = in1 * in2;
    }
}

void
INT_dense_maximum(char **args, npy_intp *dimensions, npy_intp *steps)
{
    BINARY_DENSE_LOOP {
        npy_int in1 = *((npy_int*)ip1);
        npy_int in2 = *((npy_int*)ip2);
        *((npy_int*)ip1) = (in1 >= in2) ? in1 : in2;
    }
}

void
INT_dense_minimum(char **args, npy_intp *dimensions, npy_intp *steps)
{
    BINARY_DENSE_LOOP {
        npy_int in1 = *((npy_int*)ip1);
        npy_int in2 = *((npy_int*)ip2);
        *((npy_int*)ip1) = (in1 <= in2) ? in1 : in2;
    }
}

/*
void
INT_dense_and(char **args, npy_intp *dimensions, npy_intp *steps)
{
    BINARY_DENSE_LOOP {
        npy_int in1 = *((npy_int*)ip1);
        npy_int in2 = *((npy_int*)ip2);
        *((npy_int*)ip1) = in1 and in2;
    }
}

void
INT_dense_or(char **args, npy_intp *dimensions, npy_intp *steps)
{
    BINARY_DENSE_LOOP {
        npy_int in1 = *((npy_int*)ip1);
        npy_int in2 = *((npy_int*)ip2);
        *((npy_int*)ip1) = in1 or in2;
    }
}

void
INT_dense_xor(char **args, npy_intp *dimensions, npy_intp *steps)
{
    BINARY_DENSE_LOOP {
        npy_int in1 = *((npy_int*)ip1);
        npy_int in2 = *((npy_int*)ip2);
        *((npy_int*)ip1) = in1 xor in2;
    }
}
*/

void
UINT_dense_replace(char **args, npy_intp *dimensions, npy_intp *steps)
{
    BINARY_DENSE_LOOP {
        npy_uint in2 = *((npy_uint*)ip2);
        *((npy_uint*)ip1) = in2;
    }
}

void
UINT_dense_add(char **args, npy_intp *dimensions, npy_intp *steps)
{
    BINARY_DENSE_LOOP {
        npy_uint in1 = *((npy_uint*)ip1);
        npy_uint in2 = *((npy_uint*)ip2);
        *((npy_uint*)ip1) = in1 + in2;
    }
}

void
UINT_dense_multiply(char **args, npy_intp *dimensions, npy_intp *steps)
{
    BINARY_DENSE_LOOP {
        npy_uint in1 = *((npy_uint*)ip1);
        npy_uint in2 = *((npy_uint*)ip2);
        *((npy_uint*)ip1) = in1 * in2;
    }
}

void
UINT_dense_maximum(char **args, npy_intp *dimensions, npy_intp *steps)
{
    BINARY_DENSE_LOOP {
        npy_uint in1 = *((npy_uint*)ip1);
        npy_uint in2 = *((npy_uint*)ip2);
        *((npy_uint*)ip1) = (in1 >= in2) ? in1 : in2;
    }
}

void
UINT_dense_minimum(char **args, npy_intp *dimensions, npy_intp *steps)
{
    BINARY_DENSE_LOOP {
        npy_uint in1 = *((npy_uint*)ip1);
        npy_uint in2 = *((npy_uint*)ip2);
        *((npy_uint*)ip1) = (in1 <= in2) ? in1 : in2;
    }
}

/*
void
UINT_dense_and(char **args, npy_intp *dimensions, npy_intp *steps)
{
    BINARY_DENSE_LOOP {
        npy_uint in1 = *((npy_uint*)ip1);
        npy_uint in2 = *((npy_uint*)ip2);
        *((npy_uint*)ip1) = in1 and in2;
    }
}

void
UINT_dense_or(char **args, npy_intp *dimensions, npy_intp *steps)
{
    BINARY_DENSE_LOOP {
        npy_uint in1 = *((npy_uint*)ip1);
        npy_uint in2 = *((npy_uint*)ip2);
        *((npy_uint*)ip1) = in1 or in2;
    }
}

void
UINT_dense_xor(char **args, npy_intp *dimensions, npy_intp *steps)
{
    BINARY_DENSE_LOOP {
        npy_uint in1 = *((npy_uint*)ip1);
        npy_uint in2 = *((npy_uint*)ip2);
        *((npy_uint*)ip1) = in1 xor in2;
    }
}
*/

void
LONGLONG_dense_replace(char **args, npy_intp *dimensions, npy_intp *steps)
{
    BINARY_DENSE_LOOP {
        npy_longlong in2 = *((npy_longlong*)ip2);
        *((npy_longlong*)ip1) = in2;
    }
}

void
LONGLONG_dense_add(char **args, npy_intp *dimensions, npy_intp *steps)
{
    BINARY_DENSE_LOOP {
        npy_longlong in1 = *((npy_longlong*)ip1);
        npy_longlong in2 = *((npy_longlong*)ip2);
        *((npy_longlong*)ip1) = in1 + in2;
    }
}

void
LONGLONG_dense_multiply(char **args, npy_intp *dimensions, npy_intp *steps)
{
    BINARY_DENSE_LOOP {
        npy_longlong in1 = *((npy_longlong*)ip1);
        npy_longlong in2 = *((npy_longlong*)ip2);
        *((npy_longlong*)ip1) = in1 * in2;
    }
}

void
LONGLONG_dense_maximum(char **args, npy_intp *dimensions, npy_intp *steps)
{
    BINARY_DENSE_LOOP {
        npy_longlong in1 = *((npy_longlong*)ip1);
        npy_longlong in2 = *((npy_longlong*)ip2);
        *((npy_longlong*)ip1) = (in1 >= in2) ? in1 : in2;
    }
}

void
LONGLONG_dense_minimum(char **args, npy_intp *dimensions, npy_intp *steps)
{
    BINARY_DENSE_LOOP {
        npy_longlong in1 = *((npy_longlong*)ip1);
        npy_longlong in2 = *((npy_longlong*)ip2);
        *((npy_longlong*)ip1) = (in1 <= in2) ? in1 : in2;
    }
}

/*
void
LONGLONG_dense_and(char **args, npy_intp *dimensions, npy_intp *steps)
{
    BINARY_DENSE_LOOP {
        npy_longlong in1 = *((npy_longlong*)ip1);
        npy_longlong in2 = *((npy_longlong*)ip2);
        *((npy_longlong*)ip1) = in1 and in2;
    }
}

void
LONGLONG_dense_or(char **args, npy_intp *dimensions, npy_intp *steps)
{
    BINARY_DENSE_LOOP {
        npy_longlong in1 = *((npy_longlong*)ip1);
        npy_longlong in2 = *((npy_longlong*)ip2);
        *((npy_longlong*)ip1) = in1 or in2;
    }
}

void
LONGLONG_dense_xor(char **args, npy_intp *dimensions, npy_intp *steps)
{
    BINARY_DENSE_LOOP {
        npy_longlong in1 = *((npy_longlong*)ip1);
        npy_longlong in2 = *((npy_longlong*)ip2);
        *((npy_longlong*)ip1) = in1 xor in2;
    }
}
*/

void
ULONGLONG_dense_replace(char **args, npy_intp *dimensions, npy_intp *steps)
{
    BINARY_DENSE_LOOP {
        npy_ulonglong in2 = *((npy_ulonglong*)ip2);
        *((npy_ulonglong*)ip1) = in2;
    }
}

void
ULONGLONG_dense_add(char **args, npy_intp *dimensions, npy_intp *steps)
{
    BINARY_DENSE_LOOP {
        npy_ulonglong in1 = *((npy_ulonglong*)ip1);
        npy_ulonglong in2 = *((npy_ulonglong*)ip2);
        *((npy_ulonglong*)ip1) = in1 + in2;
    }
}

void
ULONGLONG_dense_multiply(char **args, npy_intp *dimensions, npy_intp *steps)
{
    BINARY_DENSE_LOOP {
        npy_ulonglong in1 = *((npy_ulonglong*)ip1);
        npy_ulonglong in2 = *((npy_ulonglong*)ip2);
        *((npy_ulonglong*)ip1) = in1 * in2;
    }
}

void
ULONGLONG_dense_maximum(char **args, npy_intp *dimensions, npy_intp *steps)
{
    BINARY_DENSE_LOOP {
        npy_ulonglong in1 = *((npy_ulonglong*)ip1);
        npy_ulonglong in2 = *((npy_ulonglong*)ip2);
        *((npy_ulonglong*)ip1) = (in1 >= in2) ? in1 : in2;
    }
}

void
ULONGLONG_dense_minimum(char **args, npy_intp *dimensions, npy_intp *steps)
{
    BINARY_DENSE_LOOP {
        npy_ulonglong in1 = *((npy_ulonglong*)ip1);
        npy_ulonglong in2 = *((npy_ulonglong*)ip2);
        *((npy_ulonglong*)ip1) = (in1 <= in2) ? in1 : in2;
    }
}

/*
void
ULONGLONG_dense_and(char **args, npy_intp *dimensions, npy_intp *steps)
{
    BINARY_DENSE_LOOP {
        npy_ulonglong in1 = *((npy_ulonglong*)ip1);
        npy_ulonglong in2 = *((npy_ulonglong*)ip2);
        *((npy_ulonglong*)ip1) = in1 and in2;
    }
}

void
ULONGLONG_dense_or(char **args, npy_intp *dimensions, npy_intp *steps)
{
    BINARY_DENSE_LOOP {
        npy_ulonglong in1 = *((npy_ulonglong*)ip1);
        npy_ulonglong in2 = *((npy_ulonglong*)ip2);
        *((npy_ulonglong*)ip1) = in1 or in2;
    }
}

void
ULONGLONG_dense_xor(char **args, npy_intp *dimensions, npy_intp *steps)
{
    BINARY_DENSE_LOOP {
        npy_ulonglong in1 = *((npy_ulonglong*)ip1);
        npy_ulonglong in2 = *((npy_ulonglong*)ip2);
        *((npy_ulonglong*)ip1) = in1 xor in2;
    }
}
*/

void
FLOAT_dense_replace(char **args, npy_intp *dimensions, npy_intp *steps)
{
    BINARY_DENSE_LOOP {
        npy_float in2 = *((npy_float*)ip2);
        *((npy_float*)ip1) = in2;
    }
}

void
FLOAT_dense_add(char **args, npy_intp *dimensions, npy_intp *steps)
{
    BINARY_DENSE_LOOP {
        npy_float in1 = *((npy_float*)ip1);
        npy_float in2 = *((npy_float*)ip2);
        *((npy_float*)ip1) = in1 + in2;
    }
}

void
FLOAT_dense_multiply(char **args, npy_intp *dimensions, npy_intp *steps)
{
    BINARY_DENSE_LOOP {
        npy_float in1 = *((npy_float*)ip1);
        npy_float in2 = *((npy_float*)ip2);
        *((npy_float*)ip1) = in1 * in2;
    }
}

void
FLOAT_dense_maximum(char **args, npy_intp *dimensions, npy_intp *steps)
{
    BINARY_DENSE_LOOP {
        npy_float in1 = *((npy_float*)ip1);
        npy_float in2 = *((npy_float*)ip2);
        *((npy_float*)ip1) = (in1 >= in2) ? in1 : in2;
    }
}

void
FLOAT_dense_minimum(char **args, npy_intp *dimensions, npy_intp *steps)
{
    BINARY_DENSE_LOOP {
        npy_float in1 = *((npy_float*)ip1);
        npy_float in2 = *((npy_float*)ip2);
        *((npy_float*)ip1) = (in1 <= in2) ? in1 : in2;
    }
}

/*
void
FLOAT_dense_and(char **args, npy_intp *dimensions, npy_intp *steps)
{
    BINARY_DENSE_LOOP {
        npy_float in1 = *((npy_float*)ip1);
        npy_float in2 = *((npy_float*)ip2);
        *((npy_float*)ip1) = in1 and in2;
    }
}

void
FLOAT_dense_or(char **args, npy_intp *dimensions, npy_intp *steps)
{
    BINARY_DENSE_LOOP {
        npy_float in1 = *((npy_float*)ip1);
        npy_float in2 = *((npy_float*)ip2);
        *((npy_float*)ip1) = in1 or in2;
    }
}

void
FLOAT_dense_xor(char **args, npy_intp *dimensions, npy_intp *steps)
{
    BINARY_DENSE_LOOP {
        npy_float in1 = *((npy_float*)ip1);
        npy_float in2 = *((npy_float*)ip2);
        *((npy_float*)ip1) = in1 xor in2;
    }
}
*/

void
DOUBLE_dense_replace(char **args, npy_intp *dimensions, npy_intp *steps)
{
    BINARY_DENSE_LOOP {
        npy_double in2 = *((npy_double*)ip2);
        *((npy_double*)ip1) = in2;
    }
}

void
DOUBLE_dense_add(char **args, npy_intp *dimensions, npy_intp *steps)
{
    BINARY_DENSE_LOOP {
        npy_double in1 = *((npy_double*)ip1);
        npy_double in2 = *((npy_double*)ip2);
        *((npy_double*)ip1) = in1 + in2;
    }
}

void
DOUBLE_dense_multiply(char **args, npy_intp *dimensions, npy_intp *steps)
{
    BINARY_DENSE_LOOP {
        npy_double in1 = *((npy_double*)ip1);
        npy_double in2 = *((npy_double*)ip2);
        *((npy_double*)ip1) = in1 * in2;
    }
}

void
DOUBLE_dense_maximum(char **args, npy_intp *dimensions, npy_intp *steps)
{
    BINARY_DENSE_LOOP {
        npy_double in1 = *((npy_double*)ip1);
        npy_double in2 = *((npy_double*)ip2);
        *((npy_double*)ip1) = (in1 >= in2) ? in1 : in2;
    }
}

void
DOUBLE_dense_minimum(char **args, npy_intp *dimensions, npy_intp *steps)
{
    BINARY_DENSE_LOOP {
        npy_double in1 = *((npy_double*)ip1);
        npy_double in2 = *((npy_double*)ip2);
        *((npy_double*)ip1) = (in1 <= in2) ? in1 : in2;
    }
}

/*
void
DOUBLE_dense_and(char **args, npy_intp *dimensions, npy_intp *steps)
{
    BINARY_DENSE_LOOP {
        npy_double in1 = *((npy_double*)ip1);
        npy_double in2 = *((npy_double*)ip2);
        *((npy_double*)ip1) = in1 and in2;
    }
}

void
DOUBLE_dense_or(char **args, npy_intp *dimensions, npy_intp *steps)
{
    BINARY_DENSE_LOOP {
        npy_double in1 = *((npy_double*)ip1);
        npy_double in2 = *((npy_double*)ip2);
        *((npy_double*)ip1) = in1 or in2;
    }
}

void
DOUBLE_dense_xor(char **args, npy_intp *dimensions, npy_intp *steps)
{
    BINARY_DENSE_LOOP {
        npy_double in1 = *((npy_double*)ip1);
        npy_double in2 = *((npy_double*)ip2);
        *((npy_double*)ip1) = in1 xor in2;
    }
}
*/


typedef void (*dense_reducer)(char**, npy_intp*, npy_intp*);
dense_reducer dense_replace_functions[] = {BOOL_dense_replace, INT_dense_replace, UINT_dense_replace, LONGLONG_dense_replace, ULONGLONG_dense_replace, FLOAT_dense_replace, DOUBLE_dense_replace, NULL};
dense_reducer dense_add_functions[] = {BOOL_dense_add, INT_dense_add, UINT_dense_add, LONGLONG_dense_add, ULONGLONG_dense_add, FLOAT_dense_add, DOUBLE_dense_add, NULL};
dense_reducer dense_multiply_functions[] = {BOOL_dense_multiply, INT_dense_multiply, UINT_dense_multiply, LONGLONG_dense_multiply, ULONGLONG_dense_multiply, FLOAT_dense_multiply, DOUBLE_dense_multiply, NULL};
dense_reducer dense_maximum_functions[] = {BOOL_dense_maximum, INT_dense_maximum, UINT_dense_maximum, LONGLONG_dense_maximum, ULONGLONG_dense_maximum, FLOAT_dense_maximum, DOUBLE_dense_maximum, NULL};
dense_reducer dense_minimum_functions[] = {BOOL_dense_minimum, INT_dense_minimum, UINT_dense_minimum, LONGLONG_dense_minimum, ULONGLONG_dense_minimum, FLOAT_dense_minimum, DOUBLE_dense_minimum, NULL};
//dense_reducer dense_and_functions[] = {BOOL_dense_and, INT_dense_and, UINT_dense_and, LONGLONG_dense_and, ULONGLONG_dense_and, FLOAT_dense_and, DOUBLE_dense_and, NULL};
//dense_reducer dense_or_functions[] = {BOOL_dense_or, INT_dense_or, UINT_dense_or, LONGLONG_dense_or, ULONGLONG_dense_or, FLOAT_dense_or, DOUBLE_dense_or, NULL};
//dense_reducer dense_xor_functions[] = {BOOL_dense_xor, INT_dense_xor, UINT_dense_xor, LONGLONG_dense_xor, ULONGLONG_dense_xor, FLOAT_dense_xor, DOUBLE_dense_xor, NULL};

char dense_funcs_type[] = {NPY_BOOLLTR, NPY_INTLTR, NPY_UINTLTR, NPY_LONGLONGLTR, NPY_ULONGLONGLTR, NPY_FLOATLTR, NPY_DOUBLELTR, ' '};
/* This must be sync with REDUCER enumeration */
dense_reducer* dense_functions[] = {
                                     dense_replace_functions,  /* REDUCER_REPLACE */
                                     dense_add_functions,      /* REDUCER_ADD */
                                     dense_multiply_functions, /* REDUCER_MUL */
                                     dense_maximum_functions,  /* REDUCER_MAXIMUM */
                                     dense_minimum_functions,  /* REDUCER_MINIMUM */
                                     /*dense_and_functions,      [> REDUCER_AND <]*/
                                     /*dense_or_functions,       [> REDUCER_OR <]*/
                                     /*dense_xor_functions,      [> REDUCER_XOR <]*/
                                     NULL,
                                   };

dense_reducer
select_dense_reducer(REDUCER reducer, char type)
{
    int i;

    for (i = 0; dense_funcs_type[i] != ' '; i++) {
        if (dense_funcs_type[i] == type)
            return dense_functions[reducer - REDUCER_BEGIN][i];
    }
    return NULL;
}

/* reducer(ip1[ex], ip2) */
void
slice_dense_outer_loop(CArray *ip1, CArray *ip2, CExtent *ex, REDUCER reducer)
{
    char *arrays[2] = {ip1->get_data(), ip2->get_data()};
    npy_intp continous_size, all_size;
    npy_intp inner_steps[1] = {ip1->get_strides()[ip1->get_nd() - 1]};
    int i, last_sliced_dim;

    for (i = ip1->get_nd()- 1; i >= 0; i--) {
        npy_intp dim;
       
        dim = ex->lr[i] - ex->ul[i];
        dim = (dim == 0) ? 1 : dim;
        if (dim != ip1->get_dimensions()[i]) {
            break;
        }
    }

    last_sliced_dim = i;
    if (last_sliced_dim == ip1->get_nd() - 1) {
        continous_size = ip1->get_dimensions()[ip1->get_nd() - 1];
    } else {
        continous_size = 1;
        for (i = last_sliced_dim ; i < ip1->get_nd(); i++) {
            continous_size *= ip1->get_dimensions()[i];
        }
    }

    npy_intp curr_idx[NPY_MAXDIMS], curr_pos, prev_pos = 0;
    for (i = 0; i < ip1->get_nd(); i++) {
        curr_idx[i] = ex->ul[i];
    }
    all_size = ex->size;

    dense_reducer func = select_dense_reducer(reducer, ip1->get_type());
    do {
        curr_pos = ravelled_pos(curr_idx, ex->array_shape, ip1->get_nd());
        arrays[0] += (curr_pos - prev_pos);
        func(arrays, &continous_size, inner_steps);

        for (i = last_sliced_dim; i >= 0; i++) {
            curr_idx[i] += 1;
            if (curr_idx[i] - ex->ul[i] < ex->shape[i]) {
                break;
            }
            curr_idx[i] = ex->ul[i];
        }
        prev_pos = curr_pos;
        all_size -= continous_size;
    } while(all_size > 0);
}

void
trivial_dense_outer_loop(CArray *ip1, CArray *ip2, REDUCER reducer)
{
    std::cout << __func__ << std::endl;
    char *arrays[2] = {ip1->get_data(), ip2->get_data()};
    npy_intp inner_steps[1] = {ip1->get_strides()[ip1->get_nd() - 1]};
    npy_intp size = 1;
    std::cout << __func__ << std::endl;
    for (int i = 0; i < ip1->get_nd(); i++) {
        size *= ip1->get_dimensions()[i];
    }

    dense_reducer func = select_dense_reducer(reducer, ip1->get_type());
    std::cout << __func__ << " " << (void*) func << std::endl;
    func(arrays, &size, inner_steps);
}

/**
 * Sparse array to dense array
 */

#define BINARY_BEGIN_SPARSE_LOOP \
    char *dp = args[0], *rp = args[1], *cp = args[2], *vp = args[3]; \
    npy_intp dense_row_stride = dimensions[0]; \
    npy_intp dense_col_stride = dimensions[1]; \
    npy_intp n = dimensions[2]; \
    npy_intp row_base = bases[0]; \
    npy_intp col_base = bases[1]; \
    _RP_TYPE_ row, col; \
    npy_intp i; \
    if (row_base == 0 && col_base == 0) { \
        for (i = 0 ; i < n ; i++, rp++, cp++, vp++) { \
            row = *((_RP_TYPE_*)(++rp)); \
            col = *((_RP_TYPE_*)(++cp)); \
            npy_intp pos = row * dense_row_stride + col * dense_col_stride;

#define BINARY_BEGIN_SPARSE_ELSE \
        } \
    } else { \
        for (i = 0 ; i < n ; i++, rp++, cp++, vp++) { \
            row = *((_RP_TYPE_*)(rp)) + row_base; \
            col = *((_RP_TYPE_*)(cp)) + col_base; \
            npy_intp pos = row * dense_row_stride + col * dense_col_stride;\

#define BINARY_SPARSE_LOOP_END\
        } \
    }


void
BOOL_sparse_dense_add(char **args, npy_intp *dimensions, npy_intp *bases)
{

    BINARY_BEGIN_SPARSE_LOOP
    npy_bool dense_val = *((npy_bool*)(dp + pos));
    npy_bool sparse_val = *((npy_bool*)(vp));
    *((npy_bool*)(dp + pos)) = dense_val + sparse_val;
    BINARY_BEGIN_SPARSE_ELSE
    npy_bool dense_val = *((npy_bool*)(dp + pos));
    npy_bool sparse_val = *((npy_bool*)(vp));
    *((npy_bool*)(dp + pos)) = dense_val + sparse_val;
    BINARY_SPARSE_LOOP_END
}

/*
void
BOOL_sparse_dense_or(char **args, npy_intp *dimensions, npy_intp *bases)
{
    BINARY_BEGIN_SPARSE_LOOP
    npy_bool dense_val = *((npy_bool*)(dp + pos));
    npy_bool sparse_val = *((npy_bool*)(vp));
    *((npy_bool*)(dp + pos)) = dense_val or sparse_val;
    BINARY_BEGIN_SPARSE_ELSE
    npy_bool dense_val = *((npy_bool*)(dp + pos));
    npy_bool sparse_val = *((npy_bool*)(vp));
    *((npy_bool*)(dp + pos)) = dense_val or sparse_val;
    BINARY_SPARSE_LOOP_END
}
*/

void
INT_sparse_dense_add(char **args, npy_intp *dimensions, npy_intp *bases)
{

    BINARY_BEGIN_SPARSE_LOOP
    npy_int dense_val = *((npy_int*)(dp + pos));
    npy_int sparse_val = *((npy_int*)(vp));
    *((npy_int*)(dp + pos)) = dense_val + sparse_val;
    BINARY_BEGIN_SPARSE_ELSE
    npy_int dense_val = *((npy_int*)(dp + pos));
    npy_int sparse_val = *((npy_int*)(vp));
    *((npy_int*)(dp + pos)) = dense_val + sparse_val;
    BINARY_SPARSE_LOOP_END
}

/*
void
INT_sparse_dense_or(char **args, npy_intp *dimensions, npy_intp *bases)
{
    BINARY_BEGIN_SPARSE_LOOP
    npy_int dense_val = *((npy_int*)(dp + pos));
    npy_int sparse_val = *((npy_int*)(vp));
    *((npy_int*)(dp + pos)) = dense_val or sparse_val;
    BINARY_BEGIN_SPARSE_ELSE
    npy_int dense_val = *((npy_int*)(dp + pos));
    npy_int sparse_val = *((npy_int*)(vp));
    *((npy_int*)(dp + pos)) = dense_val or sparse_val;
    BINARY_SPARSE_LOOP_END
}
*/

void
UINT_sparse_dense_add(char **args, npy_intp *dimensions, npy_intp *bases)
{

    BINARY_BEGIN_SPARSE_LOOP
    npy_uint dense_val = *((npy_uint*)(dp + pos));
    npy_uint sparse_val = *((npy_uint*)(vp));
    *((npy_uint*)(dp + pos)) = dense_val + sparse_val;
    BINARY_BEGIN_SPARSE_ELSE
    npy_uint dense_val = *((npy_uint*)(dp + pos));
    npy_uint sparse_val = *((npy_uint*)(vp));
    *((npy_uint*)(dp + pos)) = dense_val + sparse_val;
    BINARY_SPARSE_LOOP_END
}

/*
void
UINT_sparse_dense_or(char **args, npy_intp *dimensions, npy_intp *bases)
{
    BINARY_BEGIN_SPARSE_LOOP
    npy_uint dense_val = *((npy_uint*)(dp + pos));
    npy_uint sparse_val = *((npy_uint*)(vp));
    *((npy_uint*)(dp + pos)) = dense_val or sparse_val;
    BINARY_BEGIN_SPARSE_ELSE
    npy_uint dense_val = *((npy_uint*)(dp + pos));
    npy_uint sparse_val = *((npy_uint*)(vp));
    *((npy_uint*)(dp + pos)) = dense_val or sparse_val;
    BINARY_SPARSE_LOOP_END
}
*/

void
LONGLONG_sparse_dense_add(char **args, npy_intp *dimensions, npy_intp *bases)
{

    BINARY_BEGIN_SPARSE_LOOP
    npy_longlong dense_val = *((npy_longlong*)(dp + pos));
    npy_longlong sparse_val = *((npy_longlong*)(vp));
    *((npy_longlong*)(dp + pos)) = dense_val + sparse_val;
    BINARY_BEGIN_SPARSE_ELSE
    npy_longlong dense_val = *((npy_longlong*)(dp + pos));
    npy_longlong sparse_val = *((npy_longlong*)(vp));
    *((npy_longlong*)(dp + pos)) = dense_val + sparse_val;
    BINARY_SPARSE_LOOP_END
}

/*
void
LONGLONG_sparse_dense_or(char **args, npy_intp *dimensions, npy_intp *bases)
{
    BINARY_BEGIN_SPARSE_LOOP
    npy_longlong dense_val = *((npy_longlong*)(dp + pos));
    npy_longlong sparse_val = *((npy_longlong*)(vp));
    *((npy_longlong*)(dp + pos)) = dense_val or sparse_val;
    BINARY_BEGIN_SPARSE_ELSE
    npy_longlong dense_val = *((npy_longlong*)(dp + pos));
    npy_longlong sparse_val = *((npy_longlong*)(vp));
    *((npy_longlong*)(dp + pos)) = dense_val or sparse_val;
    BINARY_SPARSE_LOOP_END
}
*/

void
ULONGLONG_sparse_dense_add(char **args, npy_intp *dimensions, npy_intp *bases)
{

    BINARY_BEGIN_SPARSE_LOOP
    npy_ulonglong dense_val = *((npy_ulonglong*)(dp + pos));
    npy_ulonglong sparse_val = *((npy_ulonglong*)(vp));
    *((npy_ulonglong*)(dp + pos)) = dense_val + sparse_val;
    BINARY_BEGIN_SPARSE_ELSE
    npy_ulonglong dense_val = *((npy_ulonglong*)(dp + pos));
    npy_ulonglong sparse_val = *((npy_ulonglong*)(vp));
    *((npy_ulonglong*)(dp + pos)) = dense_val + sparse_val;
    BINARY_SPARSE_LOOP_END
}

/*
void
ULONGLONG_sparse_dense_or(char **args, npy_intp *dimensions, npy_intp *bases)
{
    BINARY_BEGIN_SPARSE_LOOP
    npy_ulonglong dense_val = *((npy_ulonglong*)(dp + pos));
    npy_ulonglong sparse_val = *((npy_ulonglong*)(vp));
    *((npy_ulonglong*)(dp + pos)) = dense_val or sparse_val;
    BINARY_BEGIN_SPARSE_ELSE
    npy_ulonglong dense_val = *((npy_ulonglong*)(dp + pos));
    npy_ulonglong sparse_val = *((npy_ulonglong*)(vp));
    *((npy_ulonglong*)(dp + pos)) = dense_val or sparse_val;
    BINARY_SPARSE_LOOP_END
}
*/

void
FLOAT_sparse_dense_add(char **args, npy_intp *dimensions, npy_intp *bases)
{

    BINARY_BEGIN_SPARSE_LOOP
    npy_float dense_val = *((npy_float*)(dp + pos));
    npy_float sparse_val = *((npy_float*)(vp));
    *((npy_float*)(dp + pos)) = dense_val + sparse_val;
    BINARY_BEGIN_SPARSE_ELSE
    npy_float dense_val = *((npy_float*)(dp + pos));
    npy_float sparse_val = *((npy_float*)(vp));
    *((npy_float*)(dp + pos)) = dense_val + sparse_val;
    BINARY_SPARSE_LOOP_END
}

/*
void
FLOAT_sparse_dense_or(char **args, npy_intp *dimensions, npy_intp *bases)
{
    BINARY_BEGIN_SPARSE_LOOP
    npy_float dense_val = *((npy_float*)(dp + pos));
    npy_float sparse_val = *((npy_float*)(vp));
    *((npy_float*)(dp + pos)) = dense_val or sparse_val;
    BINARY_BEGIN_SPARSE_ELSE
    npy_float dense_val = *((npy_float*)(dp + pos));
    npy_float sparse_val = *((npy_float*)(vp));
    *((npy_float*)(dp + pos)) = dense_val or sparse_val;
    BINARY_SPARSE_LOOP_END
}
*/

void
DOUBLE_sparse_dense_add(char **args, npy_intp *dimensions, npy_intp *bases)
{

    BINARY_BEGIN_SPARSE_LOOP
    npy_double dense_val = *((npy_double*)(dp + pos));
    npy_double sparse_val = *((npy_double*)(vp));
    *((npy_double*)(dp + pos)) = dense_val + sparse_val;
    BINARY_BEGIN_SPARSE_ELSE
    npy_double dense_val = *((npy_double*)(dp + pos));
    npy_double sparse_val = *((npy_double*)(vp));
    *((npy_double*)(dp + pos)) = dense_val + sparse_val;
    BINARY_SPARSE_LOOP_END
}

/*
void
DOUBLE_sparse_dense_or(char **args, npy_intp *dimensions, npy_intp *bases)
{
    BINARY_BEGIN_SPARSE_LOOP
    npy_double dense_val = *((npy_double*)(dp + pos));
    npy_double sparse_val = *((npy_double*)(vp));
    *((npy_double*)(dp + pos)) = dense_val or sparse_val;
    BINARY_BEGIN_SPARSE_ELSE
    npy_double dense_val = *((npy_double*)(dp + pos));
    npy_double sparse_val = *((npy_double*)(vp));
    *((npy_double*)(dp + pos)) = dense_val or sparse_val;
    BINARY_SPARSE_LOOP_END
}
*/


typedef void (*sparse_dense_reducer)(char**, npy_intp*, npy_intp*);
sparse_dense_reducer sparse_dense_add_functions[] = {BOOL_sparse_dense_add, INT_sparse_dense_add, UINT_sparse_dense_add, LONGLONG_sparse_dense_add, ULONGLONG_sparse_dense_add, FLOAT_sparse_dense_add, DOUBLE_sparse_dense_add, NULL};
//sparse_dense_reducer sparse_dense_or_functions[] = {BOOL_sparse_dense_or, INT_sparse_dense_or, UINT_sparse_dense_or, LONGLONG_sparse_dense_or, ULONGLONG_sparse_dense_or, FLOAT_sparse_dense_or, DOUBLE_sparse_dense_or, NULL};

char sparse_funcs_type[] = {NPY_BOOLLTR, NPY_INTLTR, NPY_UINTLTR, NPY_LONGLONGLTR, NPY_ULONGLONGLTR, NPY_FLOATLTR, NPY_DOUBLELTR, ' '};
/* This must be sync with REDUCER enumeration */
/* Only support add and or now. These two are trivial to implement */
sparse_dense_reducer* sparse_dense_functions[] = {
                                     NULL,                            /* REDUCER_REPLACE */
                                     sparse_dense_add_functions,      /* REDUCER_ADD */
                                     NULL,                            /* REDUCER_MUL */
                                     NULL,                            /* REDUCER_MAXIMUM */
                                     NULL,                            /* REDUCER_MINIMUM */
                                     /*NULL,                            [> REDUCER_AND <]*/
                                     /*sparse_dense_or_functions,       [> REDUCER_OR <]*/
                                     /*NULL,                            [> REDUCER_XOR <]*/
                                     NULL,
                                   };

sparse_dense_reducer
select_sparse_dense_reducer(REDUCER reducer, char type)
{
    int i;

    for (i = 0; sparse_funcs_type[i] != ' '; i++) {
        if (sparse_funcs_type[i] == type)
            return sparse_dense_functions[reducer - REDUCER_BEGIN][i];
    }
    return NULL;
}

void
sparse_dense_outer_loop(CArray *dense, CArray *sparse[], CExtent *ex, REDUCER reducer)
{
    char *arrays[4] = {dense->get_data(), sparse[0]->get_data(), sparse[1]->get_data(), sparse[2]->get_data()};
    npy_intp dimensions[3] = {dense->get_strides()[0], dense->get_strides()[1],
                              sparse[0]->get_dimensions()[0]};
    npy_intp base[2] = {ex->ul[0], ex->ul[1]};

    sparse_dense_reducer func = select_sparse_dense_reducer(reducer, dense->get_type());
    func(arrays, dimensions, base);
}

/**
 * Sparse array to sparse array
 */


