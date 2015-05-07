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
    CSliceIdx(long nd) : nd(nd) {};
    //CSliceIdx(long nd, npy_intp *dimensions);
    CSliceIdx(PyObject *slice, long nd, npy_intp *dimensions);
    ~CSliceIdx() {};

    CSlice& get_slice(int index) {return slices[index];};
    const CSlice& get_slice(int index) const {return slices[index];};
    long get_nd(void) const {return nd;};
    void set_nd(long nd) {this->nd = nd;};
    friend rpc::Marshal& operator<<(rpc::Marshal& m, const CSliceIdx& o);
    friend rpc::Marshal& operator>>(rpc::Marshal& m, CSliceIdx& o);
private:
    CSlice slices[NPY_MAXDIMS];
    long nd;
};

inline rpc::Marshal& operator <<(rpc::Marshal& m, const CSliceIdx& o) {
    m << static_cast<int64_t>(o.get_nd());
    for (int i = 0; i < o.get_nd() ; i++) {
       m << static_cast<int64_t>(o.slices[i].start);
       m << static_cast<int64_t>(o.slices[i].stop);
       m << static_cast<int64_t>(o.slices[i].step);
    }
    return m;
}

inline rpc::Marshal& operator >>(rpc::Marshal& m, CSliceIdx& o) {
    int64_t nd, start, stop, step;

    m >> nd;
    o.set_nd(static_cast<long>(nd));
    for (int i = 0; i < nd ; i++) {
       m >> start;
       m >> stop;
       m >> step;
       o.slices[i].set_data(static_cast<npy_intp>(start),
                            static_cast<npy_intp>(stop),
                            static_cast<npy_intp>(step));
    }
    return m;
}

#endif
