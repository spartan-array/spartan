#include <stdio.h>
#include <unistd.h>
#include <signal.h>

#include "rpc/client.h"
#include "rpc/server.h"
#include "demo_service.h"

using namespace rpc;
using namespace demo;

bool g_stop_flag = false;

static void signal_handler(int sig) {
    Log::info("caught signal %d, stopping server now", sig);
    g_stop_flag = true;
}

int main(int argc, char* argv[]) {
    const char* bind_addr = "0.0.0.0:1987";
    printf("usage: %s [bind_addr=%s]\n", argv[0], bind_addr);
    if (argc > 1) {
        bind_addr = argv[1];
    }

    signal(SIGPIPE, SIG_IGN);
    signal(SIGHUP, SIG_IGN);
    signal(SIGCHLD, SIG_IGN);

    signal(SIGALRM, signal_handler);
    signal(SIGINT, signal_handler);
    signal(SIGQUIT, signal_handler);
    signal(SIGTERM, signal_handler);

    poll_options opts;
    PollMgr* poll = new PollMgr(opts);
    ThreadPool* thrpool = new ThreadPool(64);
    Server svr(poll, thrpool);
    poll->release();
    thrpool->release();

    DemoService math_svc;
    svr.reg(&math_svc);
    svr.start(bind_addr);

    while (g_stop_flag == false) {
        sleep(1);
    }

    return 0;
}
