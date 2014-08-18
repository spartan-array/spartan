#include <string.h>

#include "base/all.h"

using namespace base;

TEST(threading, condvar) {
    CondVar cv;
    Mutex m;
    Log::info("will wait mutex for 2 seconds");
    m.lock();
    int ret = cv.timed_wait(m, 2.0);
    m.unlock();
    EXPECT_EQ(ret, ETIMEDOUT);
    Log::info("done waiting, return = %d, err = %s", ret, strerror(ret));

    Log::info("will wait mutex for 2 seconds (not holding locks, incorrect!)");
    ret = cv.timed_wait(m, 2.0);
    // on mac, timed_wait return EINVAL
    EXPECT_TRUE(ret == EPERM || ret == EINVAL);
    Log::info("done waiting, return = %d, err = %s", ret, strerror(ret));
}


TEST(threading, condvar_timedwait) {
    CondVar cv;
    Mutex m;
    double wait_sec = 2.3;
    Log::debug("will wait for %lf sec", wait_sec);
    Timer t;
    t.start();
    m.lock();
    cv.timed_wait(m, wait_sec);
    m.unlock();
    t.stop();
    Log::debug("actually waited for %lf sec", t.elapsed());
    EXPECT_LT(fabs(t.elapsed() - wait_sec), 0.1);
}
