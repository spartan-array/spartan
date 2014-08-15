#pragma once

#include <list>
#include <queue>
#include <functional>
#include <pthread.h>

#include "basetypes.h"
#include "misc.h"

#define Pthread_spin_init(l, pshared) verify(pthread_spin_init(l, (pshared)) == 0)
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

namespace base {

class Lockable: public NoCopy {
public:
    virtual void lock() = 0;
    virtual void unlock() = 0;
};

class SpinLock: public Lockable {
public:
    SpinLock(): locked_(false) { }
    void lock();
    void unlock() {
        __sync_lock_release(&locked_);
    }
private:
    volatile bool locked_ __attribute__((aligned (64)));
};

class Mutex: public Lockable {
public:
    Mutex() {
        pthread_mutexattr_t attr;
        pthread_mutexattr_init(&attr);
        pthread_mutexattr_settype(&attr, PTHREAD_MUTEX_ERRORCHECK);
        Pthread_mutex_init(&m_, &attr);
    }
    ~Mutex() {
        Pthread_mutex_destroy(&m_);
    }

    void lock() {
        Pthread_mutex_lock(&m_);
    }
    void unlock() {
        Pthread_mutex_unlock(&m_);
    }
private:
    friend class CondVar;
    pthread_mutex_t m_;
};

// choice between spinlock & mutex:
// * when n_thread > n_core, use mutex
// * on virtual machines, use mutex


class ScopedLock: public NoCopy {
public:
    explicit ScopedLock(Lockable* lock): m_(lock) { m_->lock(); }
    explicit ScopedLock(Lockable& lock): m_(&lock) { m_->lock(); }
    ~ScopedLock() { m_->unlock(); }
private:
    Lockable* m_;
};

class CondVar: public NoCopy {
public:
    CondVar() {
        Pthread_cond_init(&cv_, nullptr);
    }
    ~CondVar() {
        Pthread_cond_destroy(&cv_);
    }

    void wait(Mutex& m) {
        Pthread_cond_wait(&cv_, &m.m_);
    }
    void signal() {
        Pthread_cond_signal(&cv_);
    }
    void bcast() {
        Pthread_cond_broadcast(&cv_);
    }

    int timed_wait(Mutex& m, double sec);
private:
    pthread_cond_t cv_;
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
        Pthread_mutex_init(&m_, nullptr);
        Pthread_cond_init(&not_empty_, nullptr);
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

    bool try_pop(T* t) {
        bool ret = false;
        Pthread_mutex_lock(&m_);
        if (!q_->empty()) {
            ret = true;
            *t = q_->front();
            q_->pop_front();
        }
        Pthread_mutex_unlock(&m_);
        return ret;
    }

    bool try_pop_but_ignore(T* t, const T& ignore) {
        bool ret = false;
        Pthread_mutex_lock(&m_);
        if (!q_->empty() && q_->front() != ignore) {
            ret = true;
            *t = q_->front();
            q_->pop_front();
        }
        Pthread_mutex_unlock(&m_);
        return ret;
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

class ThreadPool: public RefCounted {
    int n_;
    Counter round_robin_;
    pthread_t* th_;
    Queue<std::function<void()>*>* q_;
    bool should_stop_;

    static void* start_thread_pool(void*);
    void run_thread(int id_in_pool);

protected:
    ~ThreadPool();

public:
    ThreadPool(int n = get_ncpu() * 2);

    // return 0 when queuing ok, otherwise EPERM
    int run_async(const std::function<void()>&, int queuing_channel = -1);
};

class RunLater: public RefCounted {
    typedef std::pair<double, std::function<void()>*> job_t;

    pthread_t th_;
    pthread_mutex_t m_;
    pthread_cond_t cv_;
    bool should_stop_;

    SpinLock latest_l_;
    double latest_;

    std::priority_queue<job_t, std::vector<job_t>, std::greater<job_t>> jobs_;

    static void* start_run_later(void*);
    void run_later_loop();
    void try_one_job();
public:
    RunLater();

    // return 0 when queuing ok, otherwise EPERM
    int run_later(double sec, const std::function<void()>&);

    double max_wait() const;
protected:
    ~RunLater();
};

} // namespace base
