#pragma once

#include <list>
#include <map>
#include <functional>
#include <random>

#include <sys/types.h>
#include <sys/time.h>
#include <stdarg.h>
#include <stdio.h>
#include <assert.h>
#include <pthread.h>
#include <inttypes.h>

/**
 * Use assert() when the test is only intended for debugging.
 * Use verify() when the test is crucial for both debug and release binary.
 */
#ifdef NDEBUG
#define verify(expr) do { if (!(expr)) { printf("verify failed: %s at %s, line %d", #expr, __FILE__, __LINE__); abort(); } } while (0)
#else
#define verify(expr) assert(expr)
#endif

#define Pthread_spin_init(l, pshared) verify(pthread_spin_init(l, pshared) == 0)
#define Pthread_spin_lock(l) verify(pthread_spin_lock(l) == 0)
#define Pthread_spin_unlock(l) verify(pthread_spin_unlock(l) == 0)
#define Pthread_spin_destroy(l) verify(pthread_spin_destroy(l) == 0)
#define Pthread_mutex_init(m, attr) verify(pthread_mutex_init(m, attr) == 0)
#define Pthread_mutex_lock(m) verify(pthread_mutex_lock(m) == 0)
#define Pthread_mutex_unlock(m) verify(pthread_mutex_unlock(m) == 0)
#define Pthread_mutex_destroy(m) verify(pthread_mutex_destroy(m) == 0)
#define Pthread_cond_init(c, attr) verify(pthread_cond_init(c, attr) == 0)
#define Pthread_cond_destroy(c) verify(pthread_cond_destroy(c) == 0)
#define Pthread_cond_signal(c) verify(pthread_cond_signal(c) == 0)
#define Pthread_cond_broadcast(c) verify(pthread_cond_broadcast(c) == 0)
#define Pthread_cond_wait(c, m) verify(pthread_cond_wait(c, m) == 0)
#define Pthread_create(th, attr, func, arg) verify(pthread_create(th, attr, func, arg) == 0)
#define Pthread_join(th, attr) verify(pthread_join(th, attr) == 0)

namespace rpc {

typedef int32_t i32;
typedef int64_t i64;


class NoCopy {
    NoCopy(const NoCopy&);
    const NoCopy& operator =(const NoCopy&);
protected:
    NoCopy() {
    }
    virtual ~NoCopy() = 0;
};
inline NoCopy::~NoCopy() {
}


class Lockable: public NoCopy {
public:
    virtual void lock() = 0;
    virtual void unlock() = 0;
};

class SpinLock: public Lockable {
public:
    SpinLock(): locked_(0) { }
    void lock() {
        if (!lock_state() && !__sync_lock_test_and_set(&locked_, 1)) {
            return;
        }
        int wait = 1000;
        while ((wait-- > 0) && lock_state()) {
            // spin for a short while
        }
        struct timespec t;
        t.tv_sec = 0;
        t.tv_nsec = 50000;
        while (__sync_lock_test_and_set(&locked_, 1)) {
            nanosleep(&t, NULL);
        }
    }
    void unlock() {
        __sync_lock_release(&locked_);
    }

private:

    int locked_ __attribute__((aligned (64)));

    int lock_state() const volatile {
        return locked_;
    }
};

class Mutex: public Lockable {
public:
    Mutex()         {
      pthread_mutexattr_t attr;
      pthread_mutexattr_init(&attr);
      pthread_mutexattr_settype(&attr, PTHREAD_MUTEX_ERRORCHECK);
      Pthread_mutex_init(&m_, &attr);
    }
    ~Mutex()        { Pthread_mutex_destroy(&m_); }

    void lock()     { Pthread_mutex_lock(&m_); }
    void unlock()   { Pthread_mutex_unlock(&m_); }

private:
    friend class CondVar;

    pthread_mutex_t m_;
};

// choice between spinlock & mutex:
// * when n_thread > n_core, use mutex
// * on virtual machines, use mutex


// use spinlock for short critical section
typedef SpinLock ShortLock;

// use mutex for long critical section
typedef Mutex LongLock;



class ScopedLock: public NoCopy {
public:
    explicit ScopedLock(Lockable* lock): m_(lock)   { m_->lock(); }
    // Allow pass by reference.
    explicit ScopedLock(Lockable& lock): m_(&lock)  { m_->lock(); }

    ~ScopedLock()   { m_->unlock(); }

private:
    Lockable* m_;
};

class CondVar: public NoCopy {
public:
    CondVar()               { Pthread_cond_init(&cv_, NULL); }
    ~CondVar()              { Pthread_cond_destroy(&cv_); }

    void wait(Mutex* mutex) { Pthread_cond_wait(&cv_, &(mutex->m_)); }
    void signal()           { Pthread_cond_signal(&cv_); }
    void signalAll()        { Pthread_cond_broadcast(&cv_); }

    void timedWait(Mutex* mutex, const struct timespec* timeout) {
        pthread_cond_timedwait(&cv_, &(mutex->m_), timeout);
    }

private:
    pthread_cond_t cv_;
};

class Log {
    static int level;
    static FILE* fp;
    static pthread_mutex_t m;

    static void log_v(int level, int line, const char* file, const char* fmt, va_list args);
public:

    enum {
        FATAL = 0, ERROR = 1, WARN = 2, INFO = 3, DEBUG = 4
    };

    static void set_file(FILE* fp);
    static void set_level(int level);

    static void log(int level, int line, const char* file, const char* fmt, ...);

    static void fatal(int line, const char* file, const char* fmt, ...);
    static void error(int line, const char* file, const char* fmt, ...);
    static void warn(int line, const char* file, const char* fmt, ...);
    static void info(int line, const char* file, const char* fmt, ...);
    static void debug(int line, const char* file, const char* fmt, ...);

    static void fatal(const char* fmt, ...);
    static void error(const char* fmt, ...);
    static void warn(const char* fmt, ...);
    static void info(const char* fmt, ...);
    static void debug(const char* fmt, ...);
};

#define Log_debug(msg, ...) ::rpc::Log::debug(__LINE__, __FILE__, msg, ## __VA_ARGS__)
#define Log_info(msg, ...) ::rpc::Log::info(__LINE__, __FILE__, msg, ## __VA_ARGS__)
#define Log_warn(msg, ...) ::rpc::Log::warn(__LINE__, __FILE__, msg, ## __VA_ARGS__)
#define Log_error(msg, ...) ::rpc::Log::error(__LINE__, __FILE__, msg, ## __VA_ARGS__)
#define Log_fatal(msg, ...) ::rpc::Log::fatal(__LINE__, __FILE__, msg, ## __VA_ARGS__)


/**
 * Note: All sub class of RefCounted *MUST* have protected destructor!
 * This prevents accidentally deleting the object.
 * You are only allowed to cleanup with release() call.
 * This is thread safe.
 */
class RefCounted: public NoCopy {
    volatile int refcnt_;

protected:

    virtual ~RefCounted() {}

public:

    RefCounted(): refcnt_(1) {
    }

    int ref_count() {
        return refcnt_;
    }

    RefCounted* ref_copy() {
        __sync_add_and_fetch(&refcnt_, 1);
        return this;
    }

    void release() {
        int r = __sync_sub_and_fetch(&refcnt_, 1);
        verify(r >= 0);
        if (r == 0) {
            delete this;
        }
    }
};

/**
 * Thread safe queue.
 */
template<class T>
class Queue: public NoCopy {
    std::list<T>* q_;
    pthread_cond_t not_empty_;
    pthread_mutex_t m_;

public:

    Queue(): q_(new std::list<T>) {
        Pthread_mutex_init(&m_, NULL);
        Pthread_cond_init(&not_empty_, NULL);
    }

    ~Queue() {
        Pthread_cond_destroy(&not_empty_);
        Pthread_mutex_destroy(&m_);
        delete q_;
    }

    void push(const T& e) {
        Pthread_mutex_lock(&m_);
        q_->push_back(e);
        Pthread_cond_signal(&not_empty_);
        Pthread_mutex_unlock(&m_);
    }

    void pop_many(std::list<T>* out, int count) {
        Pthread_mutex_lock(&m_);
        if (q_->empty()) {
          Pthread_mutex_unlock(&m_);
          return;
        }

        while (count > 0) {
          out->push_back(q_->front());
          q_->pop_front();
          --count;
        }

        Pthread_mutex_unlock(&m_);
    }

    T pop() {
        Pthread_mutex_lock(&m_);
        while (q_->empty()) {
            Pthread_cond_wait(&not_empty_, &m_);
        }
        T e = q_->front();
        q_->pop_front();
        Pthread_mutex_unlock(&m_);
        return e;
    }
};

class Rand {
    std::mt19937 rand_;
public:
    Rand() {
        struct timeval now;
        gettimeofday(&now, NULL);
        rand_.seed(now.tv_sec + now.tv_usec + (long long) pthread_self() + (long long) this);
    }
    std::mt19937::result_type next() {
        return rand_();
    }
    std::mt19937::result_type operator() () {
        return rand_();
    }
};

class ThreadPool: public RefCounted {
    int n_;
    Rand rand_engine_;
    pthread_t* th_;
    Queue<std::function<void()>*>* q_;

    static void* start_thread_pool(void*);
    void run_thread(int id_in_pool);

protected:
    ~ThreadPool();

public:
    ThreadPool(int n = 64);

    void run_async(const std::function<void()>&);
};

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

// A microsecond precision timer based on the gettimeofday() call
// (which should be low overhead).
//
// Usage:
//
//   Timer t;
//   t.start();
//   ... event we want to clock
//   t.end();
//
//   std::cout << "elapsed time in seconds" << t.elapsed();
//
class Timer {
public:
    Timer();

    void start();
    void end();
    void reset();
    double elapsed() const;

private:
    struct timeval start_;
    struct timeval end_;
};

int set_nonblocking(int fd, bool nonblocking);

int find_open_port();
std::string get_host_name();

inline uint64_t rdtsc() {
    uint32_t hi, lo;
    __asm__ __volatile__ ("rdtsc" : "=a"(lo), "=d"(hi));
    return (((uint64_t) hi) << 32) | ((uint64_t) lo);
}


}

