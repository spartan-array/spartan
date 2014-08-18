#include <atomic>

#include "base/all.h"

using namespace base;
using namespace std;

TEST(atomic, basic_op) {
    atomic<int> sf(0);
    EXPECT_EQ(atomic_load(&sf), 0);
    int helper = 0;
    EXPECT_TRUE(atomic_compare_exchange_strong(&sf, &helper, 4));
    EXPECT_EQ(atomic_load(&sf), 4);
    helper = -3;
    EXPECT_FALSE(atomic_compare_exchange_strong(&sf, &helper, -3));
    EXPECT_EQ(atomic_load(&sf), 4);
    helper = 4;
    EXPECT_TRUE(atomic_compare_exchange_strong(&sf, &helper, -3));
    EXPECT_EQ(atomic_load(&sf), -3);
    atomic<int> sf2(atomic_load(&sf));
    EXPECT_EQ(atomic_load(&sf), -3);
    EXPECT_EQ(atomic_load(&sf2), -3);
    helper = -3;
    EXPECT_TRUE(atomic_compare_exchange_strong(&sf, &helper, 7));
    atomic_store(&sf2, atomic_load(&sf));
    EXPECT_EQ(atomic_load(&sf), 7);
    EXPECT_EQ(atomic_load(&sf2), 7);
    helper = 7;
    EXPECT_TRUE(atomic_compare_exchange_strong(&sf, &helper, 7));
}

TEST(atomic, bench) {
    atomic<int> sf(1987);
    EXPECT_EQ(atomic_load(&sf), 1987);
    Timer t;
    const int n = 100000000;
    t.start();
    int helper = 1987;
    for (int i = 0; i < n; i++) {
        EXPECT_TRUE(atomic_compare_exchange_strong(&sf, &helper, 1987));
    }
    t.stop();
    Log::debug("doing %d compare_and_swap took %.2lf sec, op/s = %s",
        n, t.elapsed(), format_decimal(n / t.elapsed()).c_str());

    t.start();
    for (int i = 0; i < n; i++) {
        EXPECT_EQ(atomic_load(&sf), 1987);
    }
    t.stop();
    Log::debug("doing %d get took %.2lf sec, op/s = %s",
        n, t.elapsed(), format_decimal(n / t.elapsed()).c_str());
}
