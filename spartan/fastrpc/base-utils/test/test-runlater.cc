#include <unistd.h>

#include "base/all.h"

using namespace base;

TEST(runlater, basic_usage) {
    RunLater* rl = new RunLater;
    Log::info("running late...");
    double defer = 2.3;
    rl->run_later(defer, [defer] {
        Log::info("deffered for %.1lf seconds!", defer);
    });
    defer = 0.1;
    rl->run_later(defer, [defer] {
        Log::info("deffered for %.1lf seconds!", defer);
    });
    Log::info("max wait = %.3lf seconds!", rl->max_wait());
    rl->release();
}


TEST(runlater, multi_waits) {
    RunLater* rl = new RunLater;
    Log::info("running late...");
    Rand r;
    for (int i = 0; i < 20; i++) {
        double defer = r.next(0, 2000) / 1000.0;
        rl->run_later(defer, [defer] {
            Log::info("deffered for %.3lf seconds!", defer);
        });
    }
    Log::info("max wait = %.3lf seconds!", rl->max_wait());
    rl->release();
}
