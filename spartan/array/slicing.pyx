cimport cython

@cython.boundscheck(False) # turn of bounds-checking for entire function
cpdef offset_slice(tuple base_ul,
                 tuple base_lr,
                 tuple other_ul,
                 tuple other_lr):
  return tuple([slice(ul-base, lr-base, None) for ul, lr, base in zip(other_ul, other_lr, base_ul)])