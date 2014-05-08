cimport cython
cimport numpy as np
from numpy cimport *

np.import_array()

# Following codes are adapted from bottleneck.argpartsort.
# Use this function to find k largest values for each row of
# an array.
# This is way faster than sort the whole array. 
def argpartsort(arr, n, axis=1):
    """
    Return indices that would partially sort an array.

    Parameters
    ----------
    arr : array_like
        Input array. If `arr` is not an array, a conversion is attempted.
    n : int
        The indices of the `n` smallest elements will appear in the first `n`
        elements of the output array along the given `axis`.
    axis : {int, None}, optional
        Axis along which the partial sort is performed. The default (axis=-1)
        is to sort along the last axis.

    Returns
    -------
    y : ndarray
        An array the same shape as the input array containing the indices
        that partially sort `arr` such that the `n` largest elements will
        appear (unordered) in the first `n` elements.
    """
    func, arr = argpartsort_selector(arr, axis)
    return func(arr, n)

def argpartsort_selector(arr, axis):
    """
    Return argpartsort function and array that matches `arr` and `axis`.
    Parameters
    ----------
    arr : array_like
        Input array. If `arr` is not an array, a conversion is attempted.
    axis : {int, None}
        Axis along which to partially sort.

    Returns
    -------
    func : function
        The argpartsort function that matches the number of dimensions and
        dtype of the input array and the axis along which you wish to partially
        sort.
    a : ndarray
        If the input array `arr` is not a ndarray, then `a` will contain the
        result of converting `arr` into a ndarray.
    """
    cdef np.ndarray a
    if type(arr) is np.ndarray:
        a = arr
    else:
        a = np.array(arr, copy=False)
    cdef tuple key
    cdef int ndim = PyArray_NDIM(a)
    cdef int dtype = PyArray_TYPE(a)
    if axis is not None:
        if axis < 0:
            axis += ndim
    else:
        a = PyArray_Ravel(a, NPY_CORDER)
        axis = 0
        ndim = 1
    key = (ndim, dtype, axis)
    func = argpartsort_dict[key]
    return func, a


@cython.boundscheck(False)
@cython.wraparound(False)
def argpartsort_2d_float64_axis1(np.ndarray[np.float64_t, ndim=2] a, int n):
    "Partial sort of 2d array with dtype=float64 along axis=1."
    cdef np.npy_intp i, j = 0, l, r, k = n-1, itmp
    cdef np.float64_t x, tmp

    # Sort original array, no copy.
    cdef np.ndarray[np.float64_t, ndim=2] b = a #PyArray_Copy(a)
    cdef Py_ssize_t i0, i1
    cdef np.npy_intp *dim
    dim = PyArray_DIMS(a)
    cdef Py_ssize_t n0 = dim[0]
    cdef Py_ssize_t n1 = dim[1]
    cdef np.npy_intp *dims = [n0, n1]
    cdef np.ndarray[np.intp_t, ndim=2] y = PyArray_EMPTY(2, dims,
		NPY_INTP, 0) 
    
    for i0 in range(n0):
        for i1 in range(n1):
            y[i0, i1] = i1
    if n1 == 0:
        return y
    if (n < 1) or (n > n1):
        raise ValueError("error")

    with nogil:
      for i0 in range(n0):
          l = 0
          r = n1 - 1
          while l < r:
              x = b[i0, k]
              i = l
              j = r
              while 1:
                  while b[i0, i] > x: i += 1
                  while x > b[i0, j]: j -= 1
                  if i <= j:
                      tmp = b[i0, i]
                      b[i0, i] = b[i0, j]
                      b[i0, j] = tmp
                      itmp = y[i0, i]
                      y[i0, i] = y[i0, j]
                      y[i0, j] = itmp
                      i += 1
                      j -= 1
                  if i > j: break
              if j < k: l = i
              if k < i: r = j
    return y

@cython.boundscheck(False)
@cython.wraparound(False)
def argpartsort_2d_float32_axis1(np.ndarray[np.float32_t, ndim=2] a, int n):
    "Partial sort of 2d array with dtype=float32 along axis=1."
    cdef np.npy_intp i, j = 0, l, r, k = n-1, itmp
    cdef np.float32_t x, tmp

    # Sort original array, no copy.
    cdef np.ndarray[np.float32_t, ndim=2] b = a #PyArray_Copy(a)
    cdef Py_ssize_t i0, i1
    cdef np.npy_intp *dim
    dim = PyArray_DIMS(a)
    cdef Py_ssize_t n0 = dim[0]
    cdef Py_ssize_t n1 = dim[1]
    cdef np.npy_intp *dims = [n0, n1]
    cdef np.ndarray[np.intp_t, ndim=2] y = PyArray_EMPTY(2, dims,
		NPY_INTP, 0)
    for i0 in range(n0):
        for i1 in range(n1):
            y[i0, i1] = i1
    if n1 == 0:
        return y
    if (n < 1) or (n > n1):
        raise ValueError("error")

    with nogil:
      for i0 in range(n0):
          l = 0
          r = n1 - 1
          while l < r:
              x = b[i0, k]
              i = l
              j = r
              while 1:
                  while b[i0, i] > x: i += 1
                  while x > b[i0, j]: j -= 1
                  if i <= j:
                      tmp = b[i0, i]
                      b[i0, i] = b[i0, j]
                      b[i0, j] = tmp
                      itmp = y[i0, i]
                      y[i0, i] = y[i0, j]
                      y[i0, j] = itmp
                      i += 1
                      j -= 1
                  if i > j: break
              if j < k: l = i
              if k < i: r = j
    return y

cdef dict argpartsort_dict = {}
argpartsort_dict[(2, NPY_FLOAT64, 1)] = argpartsort_2d_float64_axis1
argpartsort_dict[(2, NPY_FLOAT32, 1)] = argpartsort_2d_float32_axis1
