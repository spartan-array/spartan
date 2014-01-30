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

def millis(t1, t2):
    dt = t2 - t1
    ms = (dt.days * 24 * 60 * 60 + dt.seconds) * 1000 + dt.microseconds / 1000.0
    return ms

@cython.boundscheck(False) # turn of bounds-checking for entire function
def sparse_matmat_multiply(X not None, np.ndarray[ndim=2, dtype=DTYPE_FLT] W not None):
    """Multiply a sparse coo matrix by a dense matrix
    
    Parameters
    ----------
    X : scipy.sparse.coo_matrix
        A sparse matrix, of size N x M
    W : np.ndarray[dtype=DTYPE_FLT, ndim=2]
        A dense matrix, of size M x P.
        
    Returns
    -------
    A : scipy.sparse.coo_matrix
        A sparse matrix, of size N x P, the result of multiplying X by W.
	"""

    if X.shape[1] != W.shape[0]:
        raise ValueError('Matrices are not aligned!')
    	
    cdef np.ndarray[DTYPE_INT, ndim=1] rows = X.row
    cdef np.ndarray[DTYPE_INT, ndim=1] cols = X.col
    cdef np.ndarray[DTYPE_FLT, ndim=1] data = X.data
    
    cdef int n = rows.shape[0]
    cdef int p = W.shape[1]    
    
    size = rows.shape[0] * W.shape[1]
    
    cdef np.ndarray[DTYPE_INT] new_rows, new_cols
    cdef np.ndarray[DTYPE_FLT] new_data
    new_rows = numpy.zeros(size, dtype=numpy.int32)
    new_cols = numpy.zeros(size, dtype=numpy.int32)
    new_data = numpy.zeros(size, dtype=numpy.float32)

    cdef int i,j,k

    for j in range(p):
        k = j * n
        for i in range(n):
            new_rows[k+i] = rows[i]
            new_cols[k+i] = j
            new_data[k+i] = data[i] * W[cols[i], j]
            
    return scipy.sparse.coo_matrix((new_data, (new_rows, new_cols)), shape=(X.shape[0], W.shape[1]))
 
@cython.boundscheck(False) # turn of bounds-checking for entire function   
def sparse_matvec_multiply(X not None, np.ndarray[ndim=1, dtype=DTYPE_FLT] W not None):
    """Multiply a sparse coo matrix by a dense vector
    
    Parameters
    ----------
    X : scipy.sparse.coo_matrix
        A sparse matrix, of size N x M
    W : np.ndarray[dtype=float64_t, ndim=1]
        A dense vector, of size M.
        
    Returns
    -------
    A : dictionary {index: value}
        A sparse vector, the result of multiplying X by W.
    """

    if X.shape[1] != W.shape[0]:
        raise ValueError('Matrices are not aligned!')
    	
    cdef np.ndarray[DTYPE_INT, ndim=1] rows = X.row
    cdef np.ndarray[DTYPE_INT, ndim=1] cols = X.col
    cdef np.ndarray[DTYPE_FLT, ndim=1] data = X.data
    
    #cdef unordered_map[DTYPE_INT, DTYPE_FLT] new_data
    cdef np.ndarray[DTYPE_FLT] new_data
    new_data = numpy.zeros(X.shape[0], dtype=numpy.float64)
    
    cdef int i, n = rows.shape[0]    
    for i in range(n):
        new_data[rows[i]] += data[i] * W[cols[i]]
    
    #return None
    return new_data

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
    A : dictionary {index: value}
        A sparse vector, the result of multiplying X by W.
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
    for i in range(rows.shape[0]):
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
    A : dictionary {index: value}
        A sparse vector, the result of multiplying X by W.
    """

    if X.shape[1] != W.shape[0] and W.shape[1] != 1:
        raise ValueError('Matrices are not aligned!')
    	
    cdef np.ndarray[DTYPE_INT, ndim=1] rows = X.row
    cdef np.ndarray[DTYPE_INT, ndim=1] cols = X.col
    cdef np.ndarray[DTYPE_FLT, ndim=1] data = X.data
    
    cdef unordered_map[DTYPE_INT, DTYPE_FLT] result
    result.rehash(rows.size)
    
    cdef int i   
    for i in range(rows.shape[0]):
        result[rows[i]] += data[i] * W[cols[i], 0]
    
    cdef int size = result.size()
    cdef np.ndarray[DTYPE_INT] new_rows = numpy.zeros(size, dtype=numpy.int32)
    cdef np.ndarray[DTYPE_INT] new_cols = numpy.zeros(size, dtype=numpy.int32)
    cdef np.ndarray[DTYPE_FLT] new_data = numpy.zeros(size, dtype=numpy.float32)

    cdef pair[DTYPE_INT, DTYPE_FLT] entry
    cdef unordered_map[DTYPE_INT, DTYPE_FLT].iterator iter = result.begin()
    i = 0
    while iter != result.end():
        entry = deref(iter)
        new_rows[i] = entry.first
        i = i + 1
        inc(iter)

    new_rows.sort()
    for i in range(new_rows.size):
        new_data[i] = result[new_rows[i]]
 
    return scipy.sparse.coo_matrix((new_data, (new_rows, new_cols)), shape=(X.shape[0], 1))

@cython.boundscheck(False) # turn of bounds-checking for entire function   
def dot_coo_dense_dict_unsorted(X not None, np.ndarray[ndim=2, dtype=DTYPE_FLT] W not None):
    """Multiply a sparse coo matrix by a dense vector
    
    Parameters
    ----------
    X : scipy.sparse.coo_matrix
        A sparse matrix, of size N x M
    W : np.ndarray[dtype=DTYPE_FLT, ndim=1]
        A dense vector, of size M.
        
    Returns
    -------
    A : dictionary {index: value}
        A sparse vector, the result of multiplying X by W.
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
    for i in range(rows.shape[0]):
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

    #result_list = sorted(result.iteritems(), key=lambda d:d[0])
    
    i = 0
    for (key, val) in result.iteritems():
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
    A : dictionary {index: value}
        A sparse vector, the result of multiplying X by W.
    """

    if X.shape[1] != W.shape[0] and W.shape[1] != 1:
        raise ValueError('Matrices are not aligned!')
    	
    cdef np.ndarray[DTYPE_INT, ndim=1] rows = X.row
    cdef np.ndarray[DTYPE_INT, ndim=1] cols = X.col
    cdef np.ndarray[DTYPE_FLT, ndim=1] data = X.data
    
    cdef np.ndarray[DTYPE_FLT] result
    result = numpy.zeros(X.shape[0], dtype=numpy.float32)
    
    cdef int i   
    for i in range(rows.shape[0]):
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
def multiple_slice(X not None, list slices):
    if len(slices) == 0:
        return []

    if isinstance(X, scipy.sparse.coo_matrix) and X.shape[1] == 1:
        return multiple_slice_coo(X, slices)
    elif scipy.sparse.issparse(X):
        l = []
        for (blob_id, src_slice, dst_slice) in slices:
            result = X[src_slice].tocoo()
            if result.getnnz() == 0:
                continue
            l.append((blob_id, dst_slice, result))
        return l
    else:
        l = []
        for (blob_id, src_slice, dst_slice) in slices:
            l.append((blob_id, dst_slice, X[src_slice]))
        return l

@cython.boundscheck(False) # turn of bounds-checking for entire function   
def multiple_slice_coo(X not None, list slices):

    cdef np.ndarray[DTYPE_INT, ndim=1] rows = X.row
    cdef np.ndarray[DTYPE_INT, ndim=1] cols = X.col
    cdef np.ndarray[DTYPE_FLT, ndim=1] data = X.data

    cdef list results = []
    cdef int idx = 0, end_idx
    for (blob_id, src_slice, dst_slice) in slices:
        if src_slice[0].stop <= rows[idx]:
            continue

        if src_slice[0].start > rows[-1]:
            break

        end_idx = numpy.searchsorted(rows[idx:], src_slice[0].stop) + idx
        #print src_slice[0], rows[idx], rows[end_idx-1]
        results.append((blob_id, dst_slice, scipy.sparse.coo_matrix((data[idx:end_idx], (rows[idx:end_idx]-src_slice[0].start, cols[idx:end_idx])), 
                                            shape=tuple([slice.stop-slice.start for slice in src_slice]))))

        idx = end_idx
    return results

@cython.boundscheck(False) # turn of bounds-checking for entire function   
def multiple_slice_coo_unsorted(X not None, list slices):

    cdef np.ndarray[DTYPE_INT, ndim=1] rows = X.row
    cdef np.ndarray[DTYPE_INT, ndim=1] cols = X.col
    cdef np.ndarray[DTYPE_FLT, ndim=1] data = X.data

    cdef i, idx
    cdef list slice_start = [s_slice[0].start for (bid, s_slice, d_slice) in slices]
    cdef list new_data = [[] for i in range(len(slice_start))] 
    cdef list new_rows = [[] for i in range(len(slice_start))]
    for i in range(rows.size):
        idx = numpy.searchsorted(slice_start, rows[i], side='right') - 1
        if rows[i] < slices[idx][1][0].stop:
            new_rows[idx].append(rows[i]-slice_start[idx])
            new_data[idx].append(data[i])
 
    cdef list results = []
    for i in range(len(slices)):
        results.append((slices[i][0], slices[i][2], scipy.sparse.coo_matrix((new_data[i], (new_rows[i], numpy.zeros(len(new_rows[i]), dtype=numpy.int32))), 
                                            shape=tuple([slice.stop-slice.start for slice in slices[i][1]]))))

    return results


