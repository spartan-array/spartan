// Copyright (C) 2014, Peking University
// Author: Qi Chen (chenqi871025@gmail.com)
//
// Description:
//
//
#include <string>
#include <stdio.h>
#include <signal.h>

#include "worker.h"

Worker * w;

static void signal_handler(int sig) {
    Log_info("caught signal %d, stopping server now", sig);
    w->shutdown();
}

int main(int argc, char* argv[]) {

    w = new Worker();
    rpc::Server *server = new rpc::Server();
    server->reg(w);

    int ret;
    if ((ret = server->start("0.0.0.0:1111")) == 0) {
        signal(SIGPIPE, SIG_IGN);
        signal(SIGHUP, SIG_IGN);
        signal(SIGCHLD, SIG_IGN);

        signal(SIGALRM, signal_handler);
        signal(SIGINT, signal_handler);
        signal(SIGQUIT, signal_handler);
        signal(SIGTERM, signal_handler);

        w->wait_for_shutdown();
    }

    delete server;
    delete w;

    return ret;
}

