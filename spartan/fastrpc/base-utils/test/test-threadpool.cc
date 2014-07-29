#include <unistd.h>

#include "base/all.h"

using namespace base;

TEST(threading, threadpool) {
    int n_thread = 16;
    ThreadPool* tpool = new ThreadPool(n_thread);
    int n_ops = 1000 * 1000;
    Timer t;
    t.start();
    for (int i = 0; i < n_ops; i++) {
        tpool->run_async([] {});
    }
    tpool->release();
    t.stop();
    Log::info("ThreadPool(%d) took %lf sec to execute %d nop jobs, that's %.2lf usec each",
        n_thread, t.elapsed(), n_ops, t.elapsed() * 1e6 / n_ops);
}

TEST(threading, empty_threadpool) {
    int n_thread = 32;
    int n_sec = 10;
    ThreadPool* tpool = new ThreadPool(n_thread);
    Log::info("starting ThreadPool(%d) with no jobs, wait for %d seconds. CPU usage should be low.", n_thread, n_sec);
    sleep(n_sec);
    tpool->release();
}

TEST(threading, thread_queuing_channel) {
    int n_thread = 4;
    ThreadPool* tpool = new ThreadPool(n_thread);
    for (int i = 0; i < n_thread * 2; i++) {
        tpool->run_async([i] {
            Log::debug("no queuing channel: step %d", i);
            sleep(1);
        });
    }
    for (int i = 0; i < n_thread * 2; i++) {
        const int queuing_channel = 0;
        tpool->run_async([queuing_channel, i] {
            Log::debug("with queuing channel = %d: step %d", queuing_channel, i);
            sleep(1);
        }, queuing_channel);
    }
    tpool->release();
}
