cimport numpy as np
cimport cython

ctypedef np.float32_t DTYPE_FLT
ctypedef np.int32_t DTYPE_INT

cdef public enum Reducers:
  REDUCE_ADD = 0
  REDUCE_MUL = 1
  REDUCE_NONE = 2


@cython.boundscheck(False) # turn of bounds-checking for entire function
cpdef sparse_to_dense_update(np.ndarray[ndim=2, dtype=DTYPE_FLT] target, 
                             np.ndarray[ndim=2, dtype=np.uint8_t, cast=True] mask, 
                             np.ndarray[ndim=1, dtype=DTYPE_INT] rows,
                             np.ndarray[ndim=1, dtype=DTYPE_INT] cols,
                             np.ndarray[ndim=1, dtype=DTYPE_FLT] data,
                             int reducer):
  
  cdef int i
  for i in range(rows.shape[0]):
    if reducer == REDUCE_NONE or mask[rows[i], cols[i]] == 0:
      target[rows[i], cols[i]] = data[i]
    elif reducer == REDUCE_ADD:
      target[rows[i], cols[i]] = target[rows[i], cols[i]] + data[i]
    elif reducer == REDUCE_MUL:
      target[rows[i], cols[i]] = target[rows[i], cols[i]] * data[i]
    
    mask[rows[i], cols[i]] = 1
