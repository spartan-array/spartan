#include <functional>
#include <sys/time.h>

#include "misc.h"
#include "threading.h"

using namespace std;

namespace base {

void SpinLock::lock() {
    if (!locked_ && !__sync_lock_test_and_set(&locked_, true)) {
        return;
    }
    int wait = 1000;
    while ((wait-- > 0) && locked_) {
        // spin for a short while
#if defined(__i386__) || defined(__x86_64__)
        asm volatile("pause");
#endif
    }
    struct timespec t;
    t.tv_sec = 0;
    t.tv_nsec = 50000;
    while (__sync_lock_test_and_set(&locked_, true)) {
        nanosleep(&t, nullptr);
    }
}

int CondVar::timed_wait(Mutex& m, double sec) {
    int full_sec = (int) sec;
    int nsec = int((sec - full_sec) * 1000 * 1000 * 1000);
    struct timeval tv;
    gettimeofday(&tv, nullptr);
    timespec abstime;
    abstime.tv_sec = tv.tv_sec + full_sec;
    abstime.tv_nsec = tv.tv_usec * 1000 + nsec;
    if (abstime.tv_nsec > 1000 * 1000 * 1000) {
        abstime.tv_nsec -= 1000 * 1000 * 1000;
        abstime.tv_sec += 1;
    }
    return pthread_cond_timedwait(&cv_, &m.m_, &abstime);
}

struct start_thread_pool_args {
    ThreadPool* thrpool;
    int id_in_pool;
};

void* ThreadPool::start_thread_pool(void* args) {
    start_thread_pool_args* t_args = (start_thread_pool_args *) args;
    t_args->thrpool->run_thread(t_args->id_in_pool);
    delete t_args;
    pthread_exit(nullptr);
    return nullptr;
}

ThreadPool::ThreadPool(int n /* =... */): n_(n), should_stop_(false) {
    verify(n_ >= 0);
    th_ = new pthread_t[n_];
    q_ = new Queue<function<void()>*> [n_];

    for (int i = 0; i < n_; i++) {
        start_thread_pool_args* args = new start_thread_pool_args();
        args->thrpool = this;
        args->id_in_pool = i;
        Pthread_create(&th_[i], nullptr, ThreadPool::start_thread_pool, args);
    }
}

ThreadPool::~ThreadPool() {
    should_stop_ = true;
    for (int i = 0; i < n_; i++) {
        q_[i].push(nullptr);  // death pill
    }
    for (int i = 0; i < n_; i++) {
        Pthread_join(th_[i], nullptr);
    }
    // check if there's left over jobs
    for (int i = 0; i < n_; i++) {
        function<void()>* job;
        while (q_[i].try_pop(&job)) {
            if (job != nullptr) {
                (*job)();
            }
        }
    }
    delete[] th_;
    delete[] q_;
}

int ThreadPool::run_async(const std::function<void()>& f, int queuing_channel) {
    if (should_stop_) {
        return EPERM;
    }
    int queue_id;
    if (queuing_channel >= 0) {
        queue_id = queuing_channel % n_;
    } else {
        queue_id = round_robin_.next() % n_;
    }
    q_[queue_id].push(new function<void()>(f));
    return 0;
}

void ThreadPool::run_thread(int id_in_pool) {
    struct timespec sleep_req;
    const int min_sleep_nsec = 1000;  // 1us
    const int max_sleep_nsec = 50 * 1000;  // 50us
    sleep_req.tv_nsec = 1000;  // 1us
    sleep_req.tv_sec = 0;
    int stage = 0;

    // randomized stealing order
    int* steal_order = new int[n_];
    for (int i = 0; i < n_; i++) {
        steal_order[i] = i;
    }
    Rand r;
    for (int i = 0; i < n_ - 1; i++) {
        int j = r.next(i, n_);
        if (j != i) {
            int t = steal_order[j];
            steal_order[j] = steal_order[i];
            steal_order[i] = t;
        }
    }

    // fallback stages: try_pop -> sleep -> try_pop -> steal -> pop
    // succeed: sleep - 1
    // failure: sleep + 10
    for (;;) {
        function<void()>* job = nullptr;

        switch(stage) {
        case 0:
        case 2:
            if (q_[id_in_pool].try_pop(&job)) {
                stage = 0;
            } else {
                stage++;
            }
            break;
        case 1:
            nanosleep(&sleep_req, nullptr);
            stage++;
            break;
        case 3:
            for (int i = 0; i < n_; i++) {
                if (steal_order[i] != id_in_pool) {
                    // just don't steal other thread's death pill, otherwise they won't die
                    if (q_[steal_order[i]].try_pop_but_ignore(&job, nullptr)) {
                        stage = 0;
                        break;
                    }
                }
            }
            if (stage != 0) {
                stage++;
            }
            break;
        case 4:
            job = q_[id_in_pool].pop();
            stage = 0;
            break;
        }

        if (stage == 0) {
            if (job == nullptr) {
                break;
            }
            (*job)();
            delete job;
            sleep_req.tv_nsec = clamp(sleep_req.tv_nsec - 1000, min_sleep_nsec, max_sleep_nsec);
        } else {
            sleep_req.tv_nsec = clamp(sleep_req.tv_nsec + 1000, min_sleep_nsec, max_sleep_nsec);
        }
    }
    delete[] steal_order;
}

void* RunLater::start_run_later(void* thiz) {
    RunLater* rl = (RunLater *) thiz;
    rl->run_later_loop();
    pthread_exit(nullptr);
    return nullptr;
}

RunLater::RunLater() {
    should_stop_ = false;
    latest_ = 0.0;
    Pthread_mutex_init(&m_, nullptr);
    Pthread_cond_init(&cv_, nullptr);
    Pthread_create(&th_, nullptr, RunLater::start_run_later, this);
}

RunLater::~RunLater() {
    should_stop_ = true;

    Pthread_mutex_lock(&m_);
    jobs_.push(make_pair(0.0, nullptr)); // death pill
    Pthread_cond_signal(&cv_);
    Pthread_mutex_unlock(&m_);

    Pthread_join(th_, nullptr);
    Pthread_mutex_destroy(&m_);
    Pthread_cond_destroy(&cv_);
}

void RunLater::try_one_job() {
    Pthread_mutex_lock(&m_);
    if (!jobs_.empty()) {
        auto& j = jobs_.top();
        struct timeval now;
        gettimeofday(&now, nullptr);
        double now_f = now.tv_sec + now.tv_usec / 1000.0 / 1000.0;
        double wait = j.first - now_f;
        if (wait < 0.0) {
            if (j.second == nullptr) {
                // death pill
                jobs_.pop();
                Pthread_mutex_unlock(&m_);
                return;
            } else {
                (*j.second)();
                delete j.second;
                jobs_.pop();
            }
        } else {
            // wait for the time to execute a job
            struct timespec abstime;
            int wait_sec = (int) wait;
            int wait_nsec = (int) ((wait - wait_sec) * 1000.0 * 1000.0 * 1000.0);
            abstime.tv_sec = now.tv_sec;
            abstime.tv_nsec = now.tv_usec * 1000 + wait_nsec;
            if (abstime.tv_nsec > 1000 * 1000 * 1000) {
                abstime.tv_sec += 1;
                abstime.tv_nsec -= 1000 * 1000 * 1000;
            }
            int ret = pthread_cond_timedwait(&cv_, &m_, &abstime);
            verify(ret == ETIMEDOUT || ret == 0);
        }
    } else {
        // wait for inserting a new job
        Pthread_cond_wait(&cv_, &m_);
    }
    Pthread_mutex_unlock(&m_);
}

void RunLater::run_later_loop() {
    while (!should_stop_) {
        try_one_job();
    }

    bool done = false;
    while (!done) {
        Pthread_mutex_lock(&m_);
        if (jobs_.empty()) {
            done = true;
        }
        Pthread_mutex_unlock(&m_);
        if (!done) {
            try_one_job();
        }
    }
}

int RunLater::run_later(double sec, const std::function<void()>& f) {
    if (should_stop_) {
        return EPERM;
    }

    struct timeval now;
    gettimeofday(&now, nullptr);
    double later = now.tv_sec + now.tv_usec / 1000.0 / 1000.0;
    if (sec > 0.0) {
        later += sec;
    }

    latest_l_.lock();
    if (later > latest_) {
        latest_ = later;
    }
    latest_l_.unlock();

    Pthread_mutex_lock(&m_);
    jobs_.push(make_pair(later, new std::function<void()>(f)));
    Pthread_cond_signal(&cv_);
    Pthread_mutex_unlock(&m_);

    return 0;
}

double RunLater::max_wait() const {
    struct timeval now;
    gettimeofday(&now, nullptr);
    double now_f = now.tv_sec + now.tv_usec / 1000.0 / 1000.0;
    return max(0.0, latest_ - now_f);
}

} // namespace base
