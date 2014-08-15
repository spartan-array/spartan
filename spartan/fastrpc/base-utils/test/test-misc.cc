#include <string.h>
#include <unistd.h>

#include "base/all.h"

using namespace base;

TEST(misc, time_now_str) {
    int n = 1000 * 1000;
    char now[TIME_NOW_STR_SIZE];
    Timer t;
    t.start();
    for (int i = 0; i < n; i++) {
        time_now_str(now);
    }
    t.stop();
    EXPECT_EQ((int) strlen(now), TIME_NOW_STR_SIZE - 1);
    Log::info("time_now_str() %d times takes %lf sec, that's %.2lf usec per op",
        n, t.elapsed(), t.elapsed() * 1e6 / n);
}

TEST(misc, timer_elapsed) {
    Timer t;
    t.start();
    Log::info("timer start!");
    Log::info("timer elapsed: %lf", t.elapsed());
    usleep(300 * 1000); // 300ms
    Log::info("timer elapsed: %lf", t.elapsed());
    usleep(240 * 1000); // 240ms
    Log::info("timer stop!");
    t.stop();
    double t1 = t.elapsed();
    Log::info("timer elapsed: %lf", t1);
    usleep(500 * 1000); // 500ms
    double t2 = t.elapsed();
    Log::info("timer elapsed: %lf", t2);
    EXPECT_EQ(t1, t2);
}

TEST(misc, clamp) {
    EXPECT_EQ(clamp(5, 5, 5), 5);
    EXPECT_EQ(clamp(5, 1, 2), 2);
    EXPECT_EQ(clamp(5, 1, 1), 1);
    EXPECT_EQ(clamp(5, 7, 8), 7);
    EXPECT_EQ(clamp(5, 8, 8), 8);
    EXPECT_EQ(clamp(5, 1, 8), 5);
    EXPECT_EQ(clamp(1.0, 1.0, 8), 1.0);
    EXPECT_EQ(clamp(1.0, 2.0, 8.0), 2.0);
    EXPECT_EQ(clamp(1.0, 0.4, 0.8), 0.8);
}

TEST(misc, arraysize) {
    int a1[10];
    EXPECT_EQ(arraysize(a1), 10u);
    char a2[20];
    EXPECT_EQ(arraysize(a2), 20u);
}

TEST(misc, get_ncpu) {
    Log::debug("ncpu = %d", get_ncpu());
}