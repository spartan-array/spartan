// Copyright (C) 2014, Peking University
// Author: Qi Chen (chenqi871025@gmail.com)
//
// Description:

#ifndef CTILE_H
#define CTILE_H

#include "rpc/marshal.h"

struct CTile {
    enum {
        TYPE_EMPTY = 0,
        TYPE_DENSE = 1,
        TYPE_MASKED = 2,
        TYPE_SPARSE = 3,
    };
    int32_t id;
    std::string dtype;
    int8_t type;
    CTile(int32_t i): id(i) {}
    CTile() {}
};

inline rpc::Marshal& operator <<(rpc::Marshal& m, const CTile& o) {
    m << o.id;
    return m;
}

inline rpc::Marshal& operator >>(rpc::Marshal& m, CTile& o) {
    m >> o.id;
    return m;
}

#endif

