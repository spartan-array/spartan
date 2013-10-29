import numpy as np
cimport numpy as np
cimport cython

cdef float EPSILON = 1e-6
 
@cython.wraparound(False)
@cython.boundscheck(False)
cpdef _sgd_inner(np.ndarray[np.int64_t, ndim=1] rows, 
                np.ndarray[np.int64_t, ndim=1] cols,
                np.ndarray[np.float32_t, ndim=1] vals,
                np.ndarray[np.float32_t, ndim=2] u,
                np.ndarray[np.float32_t, ndim=2] m):
   
  cdef unsigned int i, offset, mid, u_idx, m_idx
  cdef np.float32_t guess, diff, rating
  cdef unsigned int rank = m.shape[1]
   
  for i in range(rows.shape[0]):
    offset = rows[i]
    mid = cols[i]
    rating = vals[i]
      
    u_idx = offset
    m_idx = mid
    
    guess = 0
     
    #util.log_info('INNER %d %d %d', u_idx, m_idx, rank)
    for i in range(rank):
      guess = guess + u[u_idx, i] * m[m_idx, i]
     
    diff = rating - guess
    for i in range(rank):
      u[u_idx, i] = u[u_idx, i] + diff * EPSILON
      m[m_idx, i] = m[m_idx, i] + diff * EPSILON
      
  return None



  
