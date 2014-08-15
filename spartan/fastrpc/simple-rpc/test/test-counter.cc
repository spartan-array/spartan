#include <atomic>

#include "rpc/utils.h"

using namespace rpc;
using namespace std;

class SpinLockCounter: public NoCopy {
    i64 next_;
    SpinLock l_;
public:
    SpinLockCounter(i64 start = 0) : next_(start) { }
    i64 peek_next() {
        l_.lock();
        i64 r = next_;
        l_.unlock();
        return r;
    }
    i64 next(i64 step = 1) {
        l_.lock();
        i64 r = next_;
        next_ += step;
        l_.unlock();
        return r;
    }
    void reset(i64 start = 0) {
        l_.lock();
        next_ = start;
        l_.unlock();
    }
};

class MutexCounter: public NoCopy {
    i64 next_;
    Mutex l_;
public:
    MutexCounter(i64 start = 0) : next_(start) { }
    i64 peek_next() {
        l_.lock();
        i64 r = next_;
        l_.unlock();
        return r;
    }
    i64 next(i64 step = 1) {
        l_.lock();
        i64 r = next_;
        next_ += step;
        l_.unlock();
        return r;
    }
    void reset(i64 start = 0) {
        l_.lock();
        next_ = start;
        l_.unlock();
    }
};

class AtomicCounter {
    atomic<i64> n_;
public:
    AtomicCounter(i64 start = 0): n_(start) {}
    i64 peek_next() const {
        return n_;
    }
    i64 next(i64 step = 1) {
        n_ += step;
        return n_;
    }
    void reset(i64 start = 0) {
        n_ = start;
    }
};

static void* worker_thread(void* f) {
    function<void()>* func = (function<void()> *) f;
    (*func)();
    pthread_exit(nullptr);
    return nullptr;
}

TEST(benchmarks, counter) {
    Timer tm;
    volatile int j = 0;
    int n = 1000 * 1000;
    int t = 64;
    tm.start();
    for (volatile int i = 0; i < n; i++) {
        j++;
    }
    tm.stop();
    double base = n / tm.elapsed();
    Log_info("i++ 1 thread: %.2lf/s", base);
    Counter ctr;
    tm.reset();
    tm.start();
    for (int i = 0; i < n; i++) {
        ctr.next();
    }
    tm.stop();
    Log_info("Counter 1 thread: %.2lf/s (%.4lf)", n / tm.elapsed(), n / tm.elapsed() / base);
    SpinLockCounter s_ctr;
    tm.reset();
    tm.start();
    for (int i = 0; i < n; i++) {
        s_ctr.next();
    }
    tm.stop();
    Log_info("SpinLockCounter 1 thread: %.2lf/s (%.4lf)", n / tm.elapsed(), n / tm.elapsed() / base);
    MutexCounter l_ctr;
    tm.reset();
    tm.start();
    for (int i = 0; i < n; i++) {
        l_ctr.next();
    }
    tm.stop();
    Log_info("MutexCounter 1 thread: %.2lf/s (%.4lf)", n / tm.elapsed(), n / tm.elapsed() / base);
    AtomicCounter a_ctr;
    tm.reset();
    tm.start();
    for (int i = 0; i < n; i++) {
        a_ctr.next();
    }
    tm.stop();
    Log_info("AtomicCounter 1 thread: %.2lf/s (%.4lf)", n / tm.elapsed(), n / tm.elapsed() / base);
    pthread_t* th = new pthread_t[t];
    function<void()> worker1 = [&ctr, n] {
        while (ctr.next() < 2 * n)
            ;
    };
    tm.reset();
    tm.start();
    for (int i = 0; i < t; i++) {
        Pthread_create(&th[i], nullptr, worker_thread, &worker1);
    }
    for (int i = 0; i < t; i++) {
        Pthread_join(th[i], nullptr);
    }
    tm.stop();
    Log_info("Counter %d thread: %.2lf/s (%.4lf)", t, n / tm.elapsed(), n / tm.elapsed() / base);
    function<void()> worker2 = [&s_ctr, n] {
        while (s_ctr.next() < 2 * n)
            ;
    };
    tm.reset();
    tm.start();
    for (int i = 0; i < t; i++) {
        Pthread_create(&th[i], nullptr, worker_thread, &worker2);
    }
    for (int i = 0; i < t; i++) {
        Pthread_join(th[i], nullptr);
    }
    tm.stop();
    Log_info("SpinLockCounter %d thread: %.2lf/s (%.4lf)", t, n / tm.elapsed(), n / tm.elapsed() / base);
    function<void()> worker3 = [&l_ctr, n] {
        while (l_ctr.next() < 2 * n)
            ;
    };
    tm.reset();
    tm.start();
    for (int i = 0; i < t; i++) {
        Pthread_create(&th[i], nullptr, worker_thread, &worker3);
    }
    for (int i = 0; i < t; i++) {
        Pthread_join(th[i], nullptr);
    }
    tm.stop();
    Log_info("MutexCounter %d thread: %.2lf/s (%.4lf)", t, n / tm.elapsed(), n / tm.elapsed() / base);
    function<void()> worker4 = [&a_ctr, n] {
        while (a_ctr.next() < 2 * n)
            ;
    };
    tm.reset();
    tm.start();
    for (int i = 0; i < t; i++) {
        Pthread_create(&th[i], nullptr, worker_thread, &worker4);
    }
    for (int i = 0; i < t; i++) {
        Pthread_join(th[i], nullptr);
    }
    tm.stop();
    Log_info("AtomicCounter %d thread: %.2lf/s (%.4lf)", t, n / tm.elapsed(), n / tm.elapsed() / base);
    delete[] th;
}
