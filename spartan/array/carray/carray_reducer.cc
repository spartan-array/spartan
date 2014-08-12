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

#include "carray.h"
#include "carray_reducer.h"
#include "../extent/cextent.h"

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

typedef void (*dense_reducer)(char**, npy_intp*, npy_intp*);
dense_reducer replace_functions[] = {INT_dense_replace, NULL};
dense_reducer add_functions[] = {INT_dense_add, NULL};
dense_reducer multiply_functions[] = {INT_dense_multiply, NULL};
dense_reducer maximum_functions[] = {INT_dense_maximum, NULL};
dense_reducer minimum_functions[] = {INT_dense_minimum, NULL};
dense_reducer and_functions[] = {INT_dense_and, NULL};
dense_reducer or_functions[] = {INT_dense_or, NULL};
dense_reducer xor_functions[] = {INT_dense_xor, NULL};

char dense_funcs_type[] = {NPY_INTLTR, ' '};
/* This must be sync with REDUCER enumeration */
dense_reducer* dense_functions[] = {
                                     replace_functions, add_functions, multiply_functions,
                                     maximum_functions, minimum_functions,
                                     and_functions, or_functions, xor_functions,
                                     NULL,
                                   };

dense_reducer 
select_dense_reducer(REDUCER reducer, char type)
{
    int i, j;

    for (i = 0; dense_funcs_type[i] != ' '; i++) {
        if (dense_funcs_type[i] == type)
            return dense_functions[reducer - REDUCER_BEGIN][i];
    }
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
    char *arrays[2] = {ip1->get_data(), ip2->get_data()};
    npy_intp inner_steps[1] = {ip1->get_strides()[ip1->get_nd() - 1]};
    npy_intp size = 1;
    for (int i = 0; i < ip1->get_nd(); i++) {
        size *= ip1->get_dimensions()[i];
    }

    dense_reducer func = select_dense_reducer(reducer, ip1->get_type());
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
    npy_int row, col; \
    npy_intp i; \
    if (row_base == 0 && col_base == 0) { \
        for (i = 0 ; i < n ; i++, rp++, cp++, vp++) { \
            row = *((npy_int*)(++rp)); \
            col = *((npy_int*)(++cp)); \
            npy_intp pos = row * dense_row_stride + col * dense_col_stride;

#define BINARY_BEGIN_SPARSE_ELSE \
        } \
    } else { \
        for (i = 0 ; i < n ; i++, rp++, cp++, vp++) { \
            row = *((npy_int*)(rp)) + row_base; \
            col = *((npy_int*)(cp)) + col_base; \
            npy_intp pos = row * dense_row_stride + col * dense_col_stride;\

#define BINARY_SPARSE_LOOP_END\
        } \
    }

void
INT_sparse_to_dense_add(char **args, npy_intp *dimensions, npy_intp *bases)
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

void
INT_sparse_to_dense_or(char **args, npy_intp *dimensions, npy_intp *bases)
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

typedef void (*sparse_reducer)(char**, npy_intp*, npy_intp*);
sparse_reducer sp_add_functions[] = {INT_sparse_to_dense_add, NULL};
sparse_reducer sp_or_functions[] = {INT_sparse_to_dense_or, NULL};

char sparse_funcs_type[] = {NPY_INTLTR, ' '};
/* This must be sync with REDUCER enumeration */
/* Only support add and or now. These two are trivial to implement */
sparse_reducer* sparse_functions[] = {
                                     NULL, sp_add_functions, NULL,
                                     NULL, NULL,
                                     NULL, sp_or_functions, NULL,
                                     NULL,
                                   };

sparse_reducer 
select_sparse_reducer(REDUCER reducer, char type)
{
    int i, j;

    for (i = 0; sparse_funcs_type[i] != ' '; i++) {
        if (sparse_funcs_type[i] == type)
            return sparse_functions[reducer - REDUCER_BEGIN][i];
    }
}

void
sparse_dense_outer_loop(CArray *dense, CArray *sparse[], CExtent *ex, REDUCER reducer)
{
    char *arrays[4] = {dense->get_data(), sparse[0]->get_data(), sparse[1]->get_data(), sparse[2]->get_data()};
    npy_intp dimensions[3] = {dense->get_strides()[0], dense->get_strides()[1], 
                              sparse[0]->get_dimensions()[0]};
    npy_intp base[2] = {ex->ul[0], ex->ul[1]};

    sparse_reducer func = select_sparse_reducer(reducer, dense->get_type());
    func(arrays, dimensions, base);
}

/**
 * Sparse array to sparse array
 */


