#include <string>

#include <stdio.h>
#include <unistd.h>
#include <signal.h>

#include "base/all.h"

#include "rpc/server.h"
#include "rpc/client.h"
#include "rlog/log_service_impl.h"
#include "rlog/rlog.h"

using namespace std;
using namespace rpc;
using namespace rlog;

TEST(integration, rlog) {
    // start rlog server
    RLogService *ls = new RLogServiceImpl;
    Server *server = new Server;
    server->reg(ls);

    EXPECT_EQ(server->start("0.0.0.0:8848"), 0);

    RLog::init("unittest-client", "127.0.0.1:8848");
    RLog::info("starting the demo_client");
    RLog::info("stopping the demo_client");
    RLog::finalize();

    RLog::init("demo_client", "127.0.0.1:8848");
    RLog::info("starting the demo_client again");
    for (int counter = 0; counter < 5; counter++) {
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

    // shutdown log server
    delete server;
    delete ls;
}
