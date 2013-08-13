#include <string>

#include <stdio.h>
#include <unistd.h>
#include <signal.h>

#include "rpc/server.h"
#include "rpc/client.h"
#include "log_service_impl.h"

using namespace std;
using namespace rpc;
using namespace logservice;

RLogService *g_ls = NULL;
Server *g_server = NULL;

static void signal_handler(int sig) {
    Log::info("caught signal %d, stopping server now", sig);
    delete g_server;
    delete g_ls;
    exit(0);
}

int main(int argc, char* argv[]) {
    string bind_addr = "0.0.0.0:8848";
    printf("usage: %s [bind_addr=%s]\n", argv[0], bind_addr.c_str());
    if (argc >= 2) {
        bind_addr = argv[1];
        if (bind_addr.find(":") == string::npos) {
            bind_addr = "0.0.0.0:" + bind_addr;
        }
    }

    Log::set_level(Log::INFO);

    g_ls = new RLogServiceImpl;
    g_server = new Server;
    g_server->reg(g_ls);

    if (g_server->start(bind_addr.c_str()) != 0) {
        delete g_server;
        delete g_ls;
        exit(1);
    }

    signal(SIGPIPE, SIG_IGN);
    signal(SIGHUP, SIG_IGN);
    signal(SIGCHLD, SIG_IGN);

    signal(SIGALRM, signal_handler);
    signal(SIGINT, signal_handler);
    signal(SIGQUIT, signal_handler);
    signal(SIGTERM, signal_handler);

    for (;;) {
        sleep(100);
    }

    // should not reach here
    verify(0);

    return 0;
}
