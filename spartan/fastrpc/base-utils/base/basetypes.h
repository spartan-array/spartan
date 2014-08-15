#pragma once

#include <vector>
#include <queue>
#include <random>
#include <inttypes.h>

#include "debugging.h"
#include "base/logging.h"

namespace base {

typedef int8_t i8;
typedef int16_t i16;
typedef int32_t i32;
typedef int64_t i64;

class SparseInt {
public:
    static size_t buf_size(char byte0);
    static size_t val_size(i64 val);
    static size_t dump(i32 val, char* buf);
    static size_t dump(i64 val, char* buf);
    static i32 load_i32(const char* buf);
    static i64 load_i64(const char* buf);
};

class v32 {
    i32 val_;
public:
    v32(i32 v = 0): val_(v) { }
    void set(i32 v) {
        val_ = v;
    }
    i32 get() const {
        return val_;
    }
    size_t val_size() const {
        return SparseInt::val_size(val_);
    }
};

class v64 {
    i64 val_;
public:
    v64(i64 v = 0): val_(v) { }
    void set(i64 v) {
        val_ = v;
    }
    i64 get() const {
        return val_;
    }
    size_t val_size() const {
        return SparseInt::val_size(val_);
    }
};

class NoCopy {
    MAKE_NOCOPY(NoCopy);
protected:
    NoCopy() {}
    virtual ~NoCopy() = 0;
};
inline NoCopy::~NoCopy() {}

/**
 * Note: All sub class of RefCounted *MUST* have protected destructor!
 * This prevents accidentally deleting the object.
 * You are only allowed to cleanup with release() call.
 * This is thread safe.
 */
class RefCounted: public NoCopy {
    volatile int refcnt_;
protected:
    virtual ~RefCounted() = 0;
public:
    RefCounted(): refcnt_(1) {}
    int ref_count() {
        return refcnt_;
    }
    RefCounted* ref_copy() {
        __sync_add_and_fetch(&refcnt_, 1);
        return this;
    }
    int release() {
        int r = __sync_sub_and_fetch(&refcnt_, 1);
        verify(r >= 0);
        if (r == 0) {
            delete this;
        }
        return r;
    }
};
inline RefCounted::~RefCounted() {}

class Counter: public NoCopy {
    volatile i64 next_;
public:
    Counter(i64 start = 0) : next_(start) { }
    i64 peek_next() const {
        return next_;
    }
    i64 next(i64 step = 1) {
        return __sync_fetch_and_add(&next_, step);
    }
    void reset(i64 start = 0) {
        next_ = start;
    }
};

class Timer {
public:
    Timer() { reset(); }
    void start();
    void stop();
    void reset();
    double elapsed() const;
private:
    struct timeval begin_;
    struct timeval end_;
};

class Rand: public NoCopy {
    std::mt19937 rand_;
public:
    Rand();
    std::mt19937::result_type next() {
        return rand_();
    }
    // [lower, upper)
    std::mt19937::result_type next(int lower, int upper) {
        return lower + rand_() % (upper - lower);
    }
    std::mt19937::result_type operator() () {
        return rand_();
    }
};

template<class T>
class Enumerator {
public:
    virtual ~Enumerator() {}
    virtual bool has_next() = 0;
    operator bool() {
        return this->has_next();
    }
    virtual T next() = 0;
    T operator() () {
        return this->next();
    }
};

// keep min-ordering
template<class T, class Compare = std::greater<T>>
class MergedEnumerator: public Enumerator<T> {
    struct merge_helper {
        T data;
        Enumerator<T>* src;

        merge_helper(const T& d, Enumerator<T>* s): data(d), src(s) {}

        bool operator < (const merge_helper& other) const {
            return Compare()(data, other.data);
        }
    };

    std::priority_queue<merge_helper, std::vector<merge_helper>> q_;

public:
    void add_source(Enumerator<T>* src) {
        if (src->has_next()) {
            q_.push(merge_helper(src->next(), src));
        }
    }
    bool has_next() {
        return !q_.empty();
    }
    T next() {
        const merge_helper& mh = q_.top();
        T ret = mh.data;
        Enumerator<T>* src = mh.src;
        q_.pop();
        if (src->has_next()) {
            q_.push(merge_helper(src->next(), src));
        }
        return ret;
    }
};

} // namespace base
