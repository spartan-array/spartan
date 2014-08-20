#define PY_ARRAY_UNIQUE_SYMBOL spartan_ctile_ARRAY_API
#define NO_IMPORT_ARRAY
#include "cslice.h"
static long long 
get_longlong(PyObject *o) {
    if (PyNumber_Check(o)) {
        PyObject *_long;
        long long ret;

        _long = PyNumber_Long(o);
        ret = PyLong_AsLongLong(_long);
        Py_DECREF(_long);
        return ret;
    } else {
        assert(0);
    }
    return 0;
}


#include <iostream>
CSliceIdx::CSliceIdx(PyObject *idx, int nd, npy_intp *dimensions)
{
    int i;

    //std::cout << "CSliceIdx" << std::endl;
    this->nd = nd;
    for (i = 0; i < nd; i++) {
        slices[i].start = 0;
        slices[i].stop = dimensions[i];
    }
    //std::cout << "0" << std::endl;
    //std::cout << __func__ << std::endl;
    if (!PyTuple_Check(idx)) {
        if (PySlice_Check(idx)) {
            Py_ssize_t start, stop, step, slicelength;
            PySlice_GetIndicesEx((PySliceObject*)idx, dimensions[0], 
                                 &start, &stop, &step, &slicelength);
            slices[0].start = (npy_intp) start;
            slices[0].stop = (npy_intp) stop;
        } else if (idx == Py_None) {
            ;
        } else {
            slices[0].start = (npy_intp)get_longlong(idx);
            slices[0].stop = slices[0].start + 1;
        }
    } else {
        for (i = 0; i < PyTuple_Size(idx); i++) {
            PyObject *slc = PyTuple_GET_ITEM(idx, i);
            //std::cout << "1" << std::endl;
            if (PySlice_Check(slc)) {
                //std::cout << "2" << std::endl;
                Py_ssize_t start, stop, step, slicelength;
                PySlice_GetIndicesEx((PySliceObject*)slc, dimensions[i], 
                                     &start, &stop, &step, &slicelength);
                slices[i].start = (npy_intp) start;
                slices[i].stop = (npy_intp) stop;
            } else if (idx == Py_None) {
                ;
            } else {
                slices[i].start = (npy_intp)get_longlong(slc);
                slices[i].stop = slices[i].start + 1;
            }
//            std::cout << "3" << std::endl;
        }
    }
 //   std::cout << "End CSliceIdx" << std::endl;
}
