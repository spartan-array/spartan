#include "base/all.h"

using namespace base;

TEST(threading, queue) {
    Queue<int> q;
    Counter ctr;
    Counter enq_ctr;
    Counter pop_ctr;
    int n_op = 1000 * 1000;
    int n_consumer = 8;
    int n_producer = 8;
    ThreadPool* tpool = new ThreadPool(n_consumer + n_producer);
    // consumer
    Timer t;
    t.start();
    for (int i = 0; i < n_consumer; i++) {
        tpool->run_async([&ctr, &enq_ctr, this, n_op, &q] {
            while (ctr.next() < n_op) {
                q.push(enq_ctr.next());
            }
        });
    }
    // producer
    for (int i = 0; i < n_producer; i++) {
        tpool->run_async([&q, this, &pop_ctr, n_op] {
            while (pop_ctr.next() < n_op) {
                q.pop();
            }
        });
    }
    tpool->release();
    t.stop();
    EXPECT_EQ(enq_ctr.peek_next(), n_op);
    Log::info("%d producer %d consumer, %d queue op took %lf sec, that's %.2lf usec each",
        n_producer, n_consumer, 2 * n_op, t.elapsed(), t.elapsed() * 1e6 / n_op / 2);
}
