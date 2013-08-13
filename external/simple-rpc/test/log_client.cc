#include <stdio.h>
#include <unistd.h>
#include <signal.h>

#include "rpc/server.h"
#include "rpc/client.h"
#include "logservice/rlog.h"

using namespace rpc;
using namespace logservice;

int main(int argc, char* argv[]) {
    RLog::init();
    RLog::info("starting the demo_client");
    RLog::info("stopping the demo_client");
    RLog::finalize();

    RLog::init("demo_client");
    RLog::info("starting the demo_client again");
    for (int counter = 0; counter < 10; counter++) {
        RLog::debug("demo debug message %d", counter);
        RLog::info("demo info message %d", counter);
        RLog::warn("demo warn message %d", counter);
        RLog::error("demo error message %d", counter);
        RLog::fatal("demo fatal message %d", counter);
    }

    for (int i = 0; i < 1000; i++) {
        usleep(17 * 1000);
        RLog::aggregate_qps("dummy", 17);
    }

    RLog::info("stopping the demo_client again");
    RLog::finalize();
    return 0;
}
