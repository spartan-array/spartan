#ifndef __SLICE_H__
#define __SLICE_H__
class Slice {
public:
    long long start;
    long long stop;
    long long step;

    Slice() {};
    Slice(long long start, long long stop, long long step) {
        this->start = start; 
        this->stop = stop;
        this->step = step;
    };
};
#endif
