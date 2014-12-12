#ifndef __CSLICE_H__
#define __CSLICE_H__
#include <Python.h>
#include <numpy/arrayobject.h>
#include <numpy/numpyconfig.h>
#include "base/logging.h"
#include "rpc/marshal.h"

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
    //CSliceIdx(int nd, npy_intp *dimensions);
    CSliceIdx(PyObject *slice, int nd, npy_intp *dimensions);
    ~CSliceIdx() {};

    CSlice& get_slice(int index) {return slices[index];};
    const CSlice& get_slice(int index) const {return slices[index];};
    int get_nd(void) const {return nd;};
    void set_nd(int nd) {this->nd = nd;};
    friend rpc::Marshal& operator<<(rpc::Marshal& m, const CSliceIdx& o); 
    friend rpc::Marshal& operator>>(rpc::Marshal& m, CSliceIdx& o);
private:
    CSlice slices[NPY_MAXDIMS];
    int nd;
};

inline rpc::Marshal& operator <<(rpc::Marshal& m, const CSliceIdx& o) {
    m << o.get_nd();
    for (int i = 0; i < o.get_nd() ; i++) {
       m << o.slices[i].start;
       m << o.slices[i].stop;
       m << o.slices[i].step;
    }
    return m;
}

inline rpc::Marshal& operator >>(rpc::Marshal& m, CSliceIdx& o) {
    int nd;
    npy_intp start, stop, step;

    m >> nd;
    o.set_nd(nd);
    for (int i = 0; i < nd ; i++) {
       m >> start;
       m >> stop;
       m >> step;
       o.slices[i].set_data(start, stop, step);
    }
    return m;
}

#endif
