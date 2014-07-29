#include "base/all.h"

using namespace base;

TEST(basetypes, rand_benchmark_single_thread) {
    Rand r;
    int n = 100 * 1000 * 1000;
    Timer t;
    t.start();
    for (int i = 0; i < n; i++) {
        r.next();
    }
    t.stop();
    Log::debug("Rand::next() %d ops in %lf sec (%.2lf nsec each)",
        n, t.elapsed(), t.elapsed() * 1e9 / n);
}

