#include "base/all.h"

using namespace base;

TEST(counter, single_thread_counter) {
    Counter ctr;
    EXPECT_EQ(ctr.peek_next(), 0);
    for (int i = 0; i < 100; i++) {
        EXPECT_EQ(ctr.peek_next(), i);
        EXPECT_EQ(ctr.next(), i);
    }
    ctr.reset(-100);
    EXPECT_EQ(ctr.peek_next(), -100);
    for (int i = 0; i < 100; i++) {
        EXPECT_EQ(ctr.peek_next(), i - 100);
        EXPECT_EQ(ctr.next(), i - 100);
    }
    int n = 1000 * 1000;
    Timer t;
    t.start();
    for (int i = 0; i < n; i++) {
        ctr.next();
    }
    t.stop();
    Log::info("Counter::next() %d times takes %lf seconds, that's %.2lf nsec each op",
        n, t.elapsed(), t.elapsed() * 1e9 / n);
}

TEST(counter, multi_thread_counter) {
    Counter ctr;
    int n_thread = 10;
    ThreadPool* tpool = new ThreadPool(n_thread);
    for (int i = 0; i < 1000; i++) {
        tpool->run_async([&ctr] { ctr.next(); });
    }
    tpool->release();
    EXPECT_EQ(ctr.peek_next(), 1000);

    ctr.reset(0);
    EXPECT_EQ(ctr.peek_next(), 0);
    tpool = new ThreadPool(n_thread);
    for (int i = 0; i < 1000; i++) {
        tpool->run_async([&ctr, this] {
            int ticket1 = ctr.next();
            int ticket2 = ctr.next();
            EXPECT_TRUE(ticket2 > ticket1);
        });
    }
    tpool->release();
    EXPECT_EQ(ctr.peek_next(), 2000);

    ctr.reset(0);
    EXPECT_EQ(ctr.peek_next(), 0);
    int n_per_thread = 1000 * 1000;
    tpool = new ThreadPool(n_thread);
    Timer t;
    t.start();
    for (int i = 0; i < n_thread; i++) {
        tpool->run_async([&ctr, &n_per_thread, this] {
            for (int j = 0; j < n_per_thread; j++) {
                ctr.next();
            }
        });
    }
    tpool->release();
    t.stop();
    int n = n_thread * n_per_thread;
    EXPECT_EQ(ctr.peek_next(), n);
    Log::info("Counter::next() %d times takes %lf seconds, that's %.2lf nsec each op (%d threads)",
        n, t.elapsed(), t.elapsed() * 1e9 / n, n_thread);
}
