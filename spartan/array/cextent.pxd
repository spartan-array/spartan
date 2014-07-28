from libcpp cimport bool
cdef extern from "cextent.h":
  cdef cppclass Slice:
    unsigned long long start
    unsigned long long stop
    unsigned long long step
    Slice() 
    Slice(unsigned long long start, unsigned long long stop, unsigned long long step) 

  cdef cppclass CExtent:
    unsigned long long *ul
    unsigned long long *lr
    unsigned long long *array_shape
    unsigned long long *shape
    unsigned long long size
    unsigned ndim
    bool has_array_shape

    CExtent(unsigned ndim, bool has_array_shape)
    void init_info()
    unsigned long long ravelled_pos()
    unsigned to_global(unsigned long long idx, int *axis)
    CExtent* add_dim()
    CExtent* clone()

  cdef CExtent* extent_create(unsigned long long ul[], 
                              unsigned long long lr[],
                              unsigned long long array_shape[],
                              unsigned ndim)
  cdef CExtent* extent_from_shape(unsigned long long shape[], unsigned ndim)
  cdef void unravelled_pos(unsigned long long idx, 
                           unsigned long long array_shape[], 
                           unsigned ndim, 
                           unsigned long long pos[]) # output

  cdef unsigned long long ravelled_pos(unsigned long long idx[],
                                       unsigned long long array_shape[],
                                       unsigned ndim)
  cdef bool all_nonzero_shape(unsigned long long shape[], unsigned ndim)
  cdef void find_rect(unsigned long long ravelled_ul,
                      unsigned long long ravelled_lr,
                      unsigned long long shape[],
                      unsigned ndim,
                      unsigned long long rect_ravelled_ul[], # output
                      unsigned long long rect_ravelled_lr[]) # output
  cdef CExtent* intersection(CExtent* a, CExtent* b)
  cdef CExtent* find_overlapping(CExtent* extent, CExtent* region)
  cdef CExtent* compute_slice_cy(CExtent* base, long long idx[], unsigned idx_len)
  cdef CExtent* offset_from(CExtent* base, CExtent* other)
  cdef Slice* offset_slice(CExtent* base, CExtent* other)
  cdef CExtent* from_slice_cy(long long idx[], 
                              unsigned long long shape[], 
                              unsigned ndim)
  cdef void shape_for_reduction(unsigned long long input_shape[],
                                unsigned ndim, 
                                unsigned axis,
                                unsigned long long shape[]) # oputput
  cdef CExtent* index_for_reduction(CExtent *index, int axis)
  cdef bool shapes_match(unsigned long long offset[],  unsigned long long data[], unsigned ndim)
  cdef CExtent* drop_axis(CExtent* ex, int axis)
  cdef void find_shape(CExtent **extents, int num_ex,
                       unsigned long long shape[]) # output

