// Copyright (C) 2014, Peking University
// Author: Qi Chen (chenqi871025@gmail.com)
//
// Description:
//
//
#include <string>
#include <stdio.h>
#include <signal.h>

#include "master.h"

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
    Pthread_mutex_init(&g_stop_mutex, nullptr);
    Pthread_cond_init(&g_stop_cond, nullptr);

    Master *w = new Master();
    rpc::Server *server = new rpc::Server();
    server->reg(w);

    int ret;
    if ((ret = server->start("0.0.0.0:1112")) == 0) {
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
        delete w;
    }

    Pthread_mutex_destroy(&g_stop_mutex);
    Pthread_cond_destroy(&g_stop_cond);

    return ret;
}

