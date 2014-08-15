#include <stdio.h>
#include <stdlib.h>
#include <semaphore.h>
#include <unistd.h>
#include <sys/time.h>
#include <signal.h>
#include <vector>
#include <map>

#include "base/all.h"
#include "rpc/client.h"
#include "rpc/server.h"
#include "benchmark_service.h"

using namespace benchmark;
using namespace rpc;
using namespace std;

char buffer[1024];
char* fmt(const char* fmt, ...) {
    va_list l;
    va_start(l, fmt);
    vsprintf(buffer, fmt, l);
    va_end(l);
    return buffer;
}

TEST(integration, sync_test) {
    const int n_servers = 4;

    auto poll_mgr = new PollMgr;
    auto thr_pool = new ThreadPool(8);
    auto svc = new BenchmarkService;
    ClientPool* client_pool = new ClientPool(poll_mgr);
    vector<BenchmarkProxy*> clients;
    vector<Server*> servers;

    int first_port = 1987;
    for (int i = 0; i < n_servers; ++i) {
        int port = first_port + i;
        auto server = new Server(poll_mgr, thr_pool);
        server->reg(svc);
        server->start(fmt("localhost:%d", port));
        auto client = new BenchmarkProxy(client_pool->get_client(fmt("localhost:%d", port)));

        servers.push_back(server);
        clients.push_back(client);
    }

    const int n_total_batches = 100;
    for (int i = 1; i <= n_total_batches; ++i) {
        if (i % 10 == 0) {
            Log_info("Running %d/%d batch...", i, n_total_batches);
        }
        vector<Future*> f;
        for (int j = 0; j < 10000; ++j) {
            f.push_back(clients[j % n_servers]->async_prime(j));
        }
        for (auto& fu : f) {
            fu->wait();
            fu->release();
        }
    }

    delete client_pool;
    for (auto& clnt : clients) {
        delete clnt;
    }
    for (auto& svr : servers) {
        delete svr;
    }
    delete svc;
    thr_pool->release();
    poll_mgr->release();
}
