#include <string>

#include <stdio.h>
#include <unistd.h>
#include <signal.h>

#include "rpc/server.h"
#include "rpc/client.h"
#include "log_service_impl.h"

using namespace std;
using namespace rpc;
using namespace rlog;

pthread_mutex_t g_stop_mutex;
pthread_cond_t g_stop_cond;
bool g_stop_flag = false;

static void signal_handler(int sig) {
    Log_info("caught signal %d, stopping server now", sig);
    g_stop_flag = true;
    Pthread_mutex_lock(&g_stop_mutex);
    Pthread_cond_signal(&g_stop_cond);
    Pthread_mutex_unlock(&g_stop_mutex);
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

    Pthread_mutex_init(&g_stop_mutex, nullptr);
    Pthread_cond_init(&g_stop_cond, nullptr);

    RLogService* log_service = new RLogServiceImpl;
    Server* server = new Server;
    server->reg(log_service);

    int ret;
    if ((ret = server->start(bind_addr.c_str())) == 0) {
        signal(SIGPIPE, SIG_IGN);
        signal(SIGHUP, SIG_IGN);
        signal(SIGCHLD, SIG_IGN);

        signal(SIGALRM, signal_handler);
        signal(SIGINT, signal_handler);
        signal(SIGQUIT, signal_handler);
        signal(SIGTERM, signal_handler);

        Pthread_mutex_lock(&g_stop_mutex);
        while (g_stop_flag == false) {
            Pthread_cond_wait(&g_stop_cond, &g_stop_mutex);
        }
        Pthread_mutex_unlock(&g_stop_mutex);

        delete server;
        delete log_service;
    }

    Pthread_mutex_destroy(&g_stop_mutex);
    Pthread_cond_destroy(&g_stop_cond);

    return ret;
}
