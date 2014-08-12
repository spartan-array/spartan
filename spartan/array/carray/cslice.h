#ifndef __CSLICE_H__
#define __CSLICE_H__
#include <Python.h>
#include <numpy/arrayobject.h>
#include <numpy/numpyconfig.h>

class CSlice {
public:
    npy_intp start;
    npy_intp stop;
    npy_intp step;

    CSlice() : start(0), stop(0), step(1) {};
    CSlice(npy_intp start, npy_intp stop, npy_intp step) {
        this->start = start; 
        this->stop = stop;
        this->step = step;
    };
    void set_data(npy_intp start, npy_intp stop, npy_intp step) {
        this->start = start;
        this->stop = stop;
        this->step = step;
    };
    //PyObject* to_npy();
};

class CSliceIdx {
public:
    CSliceIdx() : nd(0) {};
    CSliceIdx(int nd) : nd(nd) {};
    CSliceIdx(PyObject *slice, int nd, npy_intp *dimensions);
    ~CSliceIdx() {};

    CSlice& get_slice(int index) {return slices[index];};
    int get_nd(void) {return nd;};
private:
    CSlice slices[NPY_MAXDIMS];
    int nd;
};
#endif
