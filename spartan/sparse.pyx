# distutils: language = c++
from cython.operator cimport dereference as deref, preincrement as inc
from libcpp.pair cimport pair
from libcpp.map cimport map
from unordered_map cimport unordered_map
import numpy
import scipy.sparse
cimport numpy as np
cimport cython
from datetime import datetime

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
  cdef int size = rows.shape[0]
  with nogil:
    for i in xrange(size):
      if reducer == REDUCE_NONE or mask[rows[i], cols[i]] == 0:
        target[rows[i], cols[i]] = data[i]
        mask[rows[i], cols[i]] = 1
      elif reducer == REDUCE_ADD:
        target[rows[i], cols[i]] = target[rows[i], cols[i]] + data[i]
      elif reducer == REDUCE_MUL:
        target[rows[i], cols[i]] = target[rows[i], cols[i]] * data[i]
    
@cython.boundscheck(False) # turn of bounds-checking for entire function   
def dot_coo_dense_dict(X not None, np.ndarray[ndim=2, dtype=DTYPE_FLT] W not None):
    """Multiply a sparse coo matrix by a dense vector
    
    Parameters
    ----------
    X : scipy.sparse.coo_matrix
        A sparse matrix, of size N x M
    W : np.ndarray[dtype=DTYPE_FLT, ndim=1]
        A dense vector, of size M.
        
    Returns
    -------
    A : coo matrix, the result of multiplying X by W.
    """

    if X.shape[1] != W.shape[0] and W.shape[1] != 1:
        raise ValueError('Matrices are not aligned!')
      
    cdef np.ndarray[DTYPE_INT, ndim=1] rows = X.row
    cdef np.ndarray[DTYPE_INT, ndim=1] cols = X.col
    cdef np.ndarray[DTYPE_FLT, ndim=1] data = X.data
    
    #cdef map[DTYPE_INT, DTYPE_FLT] result
    cdef dict result = {}
    #cdef np.ndarray[DTYPE_FLT] result
    #result = numpy.zeros(X.shape[0], dtype=numpy.float64)
    
    cdef int i   
    for i in xrange(rows.shape[0]):
        #result[rows[i]] += data[i] * W[cols[i], 0]
        if not result.has_key(rows[i]):
            result[rows[i]] = data[i] * W[cols[i], 0]
        else:
            result[rows[i]] += data[i] * W[cols[i], 0]
    
    cdef int size = len(result)
    #cdef int size = result.size()
    cdef np.ndarray[DTYPE_INT] new_rows = numpy.zeros(size, dtype=numpy.int32)
    cdef np.ndarray[DTYPE_INT] new_cols = numpy.zeros(size, dtype=numpy.int32)
    cdef np.ndarray[DTYPE_FLT] new_data = numpy.zeros(size, dtype=numpy.float32)

    result_list = sorted(result.iteritems(), key=lambda d:d[0])
    
    i = 0
    for (key, val) in result_list:
        new_rows[i] = key
        new_data[i] = val
        i = i + 1

    #cdef pair[DTYPE_INT, DTYPE_FLT] entry
    #cdef map[DTYPE_INT, DTYPE_FLT].iterator iter = result.begin()
    #i = 0
    #while iter != result.end():
    #    entry = deref(iter)
    #    new_rows[i] = entry.first
    #    new_data[i] = entry.second
    #    i = i + 1
    #    inc(iter)

    return scipy.sparse.coo_matrix((new_data, (new_rows, new_cols)), shape=(X.shape[0], 1))

@cython.boundscheck(False) # turn of bounds-checking for entire function   
def dot_coo_dense_unordered_map(X not None, np.ndarray[ndim=2, dtype=DTYPE_FLT] W not None):
    """Multiply a sparse coo matrix by a dense vector
    
    Parameters
    ----------
    X : scipy.sparse.coo_matrix
        A sparse matrix, of size N x M
    W : np.ndarray[dtype=DTYPE_FLT, ndim=1]
        A dense vector, of size M.
        
    Returns
    -------
    A : coo matrix, the result of multiplying X by W.
    """

    if X.shape[1] != W.shape[0] and W.shape[1] != 1:
        raise ValueError('Matrices are not aligned!')
      
    cdef np.ndarray[DTYPE_INT, ndim=1] rows = X.row
    cdef np.ndarray[DTYPE_INT, ndim=1] cols = X.col
    cdef np.ndarray[DTYPE_FLT, ndim=1] data = X.data
    
    cdef int row_size = rows.size
    
    cdef unordered_map[DTYPE_INT, DTYPE_FLT] result
    cdef int i
     
    with nogil:
        result.rehash(row_size)
          
        for i in xrange(row_size):
            result[rows[i]] += data[i] * W[cols[i], 0]
    
    cdef int size = result.size()
    
    cdef np.ndarray[DTYPE_INT] new_rows = numpy.zeros(size, dtype=numpy.int32)
    cdef np.ndarray[DTYPE_INT] new_cols = numpy.zeros(size, dtype=numpy.int32)
    cdef np.ndarray[DTYPE_FLT] new_data = numpy.zeros(size, dtype=numpy.float32)

    cdef pair[DTYPE_INT, DTYPE_FLT] entry
    cdef unordered_map[DTYPE_INT, DTYPE_FLT].iterator iter = result.begin()
    
    with nogil:
        i = 0
        while iter != result.end():
            new_rows[i] = deref(iter).first
            i = i + 1
            inc(iter)

    new_rows.sort()
    
    with nogil:
        for i in xrange(size):
            new_data[i] = result[new_rows[i]]
 
    return scipy.sparse.coo_matrix((new_data, (new_rows, new_cols)), shape=(X.shape[0], 1))

@cython.boundscheck(False) # turn of bounds-checking for entire function   
def dot_coo_dense_vec(X not None, np.ndarray[ndim=2, dtype=DTYPE_FLT] W not None):
    """Multiply a sparse coo matrix by a dense vector
    
    Parameters
    ----------
    X : scipy.sparse.coo_matrix
        A sparse matrix, of size N x M
    W : np.ndarray[dtype=DTYPE_FLT, ndim=1]
        A dense vector, of size M.
        
    Returns
    -------
    A : coo matrix, the result of multiplying X by W.
    """

    if X.shape[1] != W.shape[0] and W.shape[1] != 1:
        raise ValueError('Matrices are not aligned!')
      
    cdef np.ndarray[DTYPE_INT, ndim=1] rows = X.row
    cdef np.ndarray[DTYPE_INT, ndim=1] cols = X.col
    cdef np.ndarray[DTYPE_FLT, ndim=1] data = X.data
    
    cdef np.ndarray[DTYPE_FLT] result
    result = numpy.zeros(X.shape[0], dtype=numpy.float32)
    
    cdef int i   
    for i in xrange(rows.shape[0]):
        result[rows[i]] += data[i] * W[cols[i], 0]
    
    cdef np.ndarray[DTYPE_INT] new_rows = result.nonzero()[0].astype(numpy.int32)
    cdef np.ndarray[DTYPE_INT] new_cols = numpy.zeros(new_rows.size, dtype=numpy.int32)
    cdef np.ndarray[DTYPE_FLT] new_data = result[new_rows]

    return scipy.sparse.coo_matrix((new_data, (new_rows, new_cols)), shape=(X.shape[0], 1))

@cython.boundscheck(False) # turn of bounds-checking for entire function   
def slice(X not None, tuple slices):
    if slices is None:
        return X

    if isinstance(X, scipy.sparse.coo_matrix) and X.shape[1] == 1:
        return slice_coo(X, slices)
    elif scipy.sparse.issparse(X):
        result = X[slices].tocoo()
        if result.getnnz() == 0:
            return None
        return result
    else:
        return X[slices]

@cython.boundscheck(False) # turn of bounds-checking for entire function   
def slice_coo(X not None, tuple slices):
 
    cdef np.ndarray[DTYPE_INT, ndim=1] rows = X.row
    cdef np.ndarray[DTYPE_INT, ndim=1] cols = X.col
    cdef np.ndarray[DTYPE_FLT, ndim=1] data = X.data
 
    cdef int row_begin = slices[0].start
    cdef int row_end = slices[0].stop
    
    cdef int idx_begin = numpy.searchsorted(rows, row_begin)
    cdef int idx_end = numpy.searchsorted(rows, row_end)

    if idx_end - idx_begin <= 0:
        return None

    return scipy.sparse.coo_matrix((data[idx_begin:idx_end], (rows[idx_begin:idx_end]-row_begin, cols[idx_begin:idx_end])), shape=tuple([slice.stop-slice.start for slice in slices]))

@cython.boundscheck(False) # turn of bounds-checking for entire function
def convert_sparse_array(array, use_getitem = True):
  if array.shape[0] > array.shape[1]:
    if not use_getitem and array.nnz < array.shape[0]:
      return array.tocoo()
    else:
      return array.tocsc()
  else:
    if not use_getitem and array.nnz < array.shape[1]:
      return array.tocoo()
    else:
      return array.tocsr()

@cython.boundscheck(False) # turn of bounds-checking for entire function   
def compute_sparse_update(data, update, slices, reducer = None):
  '''
  csr_matrix and csc_matrix can't support fancing indexing like
  data[slices] = update. This API uses hstack and vstack to implement update.
  This is a out-place update and can only support slicing with step is 1.
  '''
  if data.shape[0] > data.shape[1]:
    data = data.tocsc()
    update = update.tocsc()
    if reducer is not None:
      update = reducer(data[slices], update).tocsc()
  else:
    data = data.tocsr()
    update = update.tocsr()
    if reducer is not None:
      update = reducer(data[slices], update).tocsr()

  upper_slice = (__builtins__.slice(0, slices[0].start),
                 __builtins__.slice(0, data.shape[1]))
  midleft_slice = (__builtins__.slice(slices[0].start, slices[0].stop),
                   __builtins__.slice(0, slices[1].start))
  midright_slice = (__builtins__.slice(slices[0].start, slices[0].stop),
                    __builtins__.slice(slices[1].stop, data.shape[1]))
  lower_slice = (__builtins__.slice(slices[0].stop, data.shape[0]),
                 __builtins__.slice(0, data.shape[1]))

  if slices[1].start > 0:
    update = scipy.sparse.hstack((data[midleft_slice], update), dtype = data.dtype)
  if slices[1].stop < data.shape[1]:
    update = scipy.sparse.hstack((update, data[midright_slice]), dtype = data.dtype)
  if slices[0].start > 0:
    update = scipy.sparse.vstack((data[upper_slice], update), dtype = data.dtype)
  if slices[0].stop < data.shape[0]:
    update = scipy.sparse.vstack((update, data[lower_slice]), dtype = data.dtype)

  return update

@cython.boundscheck(False) # turn of bounds-checking for entire function
def multiple_slice(X not None, list slices):
    if len(slices) == 0:
        return []

    if isinstance(X, scipy.sparse.coo_matrix) and X.shape[1] == 1:
        return multiple_slice_coo(X, slices)
    elif scipy.sparse.issparse(X):
        l = []
        for (tile_id, src_slice, dst_slice) in slices:
            result = X[src_slice]
            if result.getnnz() == 0:
                continue
            result = convert_sparse_array(result, use_getitem = False)
            l.append((tile_id, dst_slice, result))
        return l
    else:
        l = []
        for (tile_id, src_slice, dst_slice) in slices:
            l.append((tile_id, dst_slice, X[src_slice]))
        return l

cdef DTYPE_INT searchsorted(np.ndarray[DTYPE_INT, ndim=1] rows, DTYPE_INT left, DTYPE_INT right, DTYPE_INT e):
    cdef DTYPE_INT middle
    
    while left <= right:
      middle = (left + right)/2
      if rows[middle] > e:
        right = middle - 1
      elif rows[middle] < e:
        left = middle + 1
      else:
        return middle
    return left
  
@cython.boundscheck(False) # turn of bounds-checking for entire function   
def multiple_slice_coo(X not None, list slices):

    cdef np.ndarray[DTYPE_INT, ndim=1] rows = X.row
    cdef np.ndarray[DTYPE_INT, ndim=1] cols = X.col
    cdef np.ndarray[DTYPE_FLT, ndim=1] data = X.data
    
    cdef DTYPE_INT last_idx = rows.size - 1
    
    cdef list results = []
    cdef DTYPE_INT idx = 0, end_idx
    for (tile_id, src_slice, dst_slice) in slices:
        if src_slice[0].stop <= rows[idx]:
            continue

        if src_slice[0].start > rows[-1]:
            break

        end_idx = searchsorted(rows, idx, last_idx, src_slice[0].stop)
        #end_idx = numpy.searchsorted(rows[idx:], src_slice[0].stop) + idx
        #print src_slice[0], rows[idx], rows[end_idx-1]
        results.append((tile_id, dst_slice, scipy.sparse.coo_matrix((data[idx:end_idx], (rows[idx:end_idx]-src_slice[0].start, cols[idx:end_idx])),
                                            shape=tuple([slice.stop-slice.start for slice in src_slice]))))

        idx = end_idx
    return results
