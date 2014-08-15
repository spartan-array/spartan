#include <signal.h>
#include <pthread.h>

#include <iostream>
#include <fstream>
#include <vector>
#include <string>

#include "base/all.h"
#include "rlog/rlog.h"
#include "floodtest_service.h"

using namespace std;
using namespace base;
using namespace rpc;
using namespace rlog;
using namespace floodtest;

static Counter flood_counter;
static Counter flood_udp_received;
static vector<string> flood_nodes;
static ClientPool* client_pool = nullptr;

pthread_mutex_t g_nodes_mutex = PTHREAD_MUTEX_INITIALIZER;


void FloodService::update_node_list(const std::vector<std::string>& nodes) {
    Pthread_mutex_lock(&g_nodes_mutex);
    flood_nodes = nodes;
    Pthread_mutex_unlock(&g_nodes_mutex);
}

void FloodService::flood() {
    flood_counter.next();
}

void FloodService::flood_udp() {
    flood_udp_received.next();
}



static bool g_stop_flag = false;


static void signal_handler(int sig) {
    if (g_stop_flag == false) {
        Log::info("caught signal %d, stopping server now", sig);
        g_stop_flag = true;
    } else {
        Log::info("caught signal %d for the second time, kill server now", sig);
        exit(-sig);
    }
}


static std::vector<std::string> server_list() {
    vector<string> servers;
    ifstream fin("test/floodtest-servers.txt");
    string line;
    while (getline(fin, line)) {
        if (line == "" || line[0] == '#') {
            continue;
        }
        servers.push_back(line);
    }
    verify(servers.size() > 0);
    return servers;
}


static void* flood_loop(void* args) {
    Timer tm;
    tm.start();
    i32 rpc_count = 0;
    while (!g_stop_flag) {
        Pthread_mutex_lock(&g_nodes_mutex);
        auto nodes = flood_nodes;
        Pthread_mutex_unlock(&g_nodes_mutex);

        if (nodes.size() == 0) {
            Log::debug("server list empty, will retry later");
            sleep(1);
        } else {
            vector<Future*> fu_list;
            for (auto& addr : nodes) {
                Client* clnt = client_pool->get_client(addr);
                FloodProxy proxy(clnt);
                Future* fu = proxy.async_flood();
                if (fu != nullptr) {
                    fu_list.push_back(fu);
                    rpc_count++;
                }
            }
            for (auto& fu: fu_list) {
                fu->wait();
                Future::safe_release(fu);
            }
            if (tm.elapsed() > 0.4) {
                Log::debug("RPC done: %d", rpc_count);
                RLog::aggregate_qps("flood", rpc_count);
                rpc_count = 0;
                tm.reset();
                tm.start();
            }
        }
    }
    pthread_exit(nullptr);
    return nullptr;
}


int main(int argc, char* argv[]) {
    if (argc < 2) {
        printf("usage: %s <server-id>\n", argv[0]);
        exit(1);
    }

    RLog::init();

    signal(SIGPIPE, SIG_IGN);
    signal(SIGHUP, SIG_IGN);
    signal(SIGCHLD, SIG_IGN);

    signal(SIGALRM, signal_handler);
    signal(SIGINT, signal_handler);
    signal(SIGQUIT, signal_handler);
    signal(SIGTERM, signal_handler);

    size_t server_id = atoi(argv[1]);
    verify(server_id < server_list().size());
    string bind_addr = server_list()[server_id];

    FloodService* svc = new FloodService;
    PollMgr* poll = new PollMgr(1);
    ThreadPool* thrpool = new ThreadPool(1);
    Server *server = new Server(poll, thrpool);
    server->reg(svc);

    int ret_code = 0;
    if (server->start(bind_addr.c_str()) != 0) {
        ret_code = 1;
        g_stop_flag = true;
    }

    pthread_t flood_th;
    if (!g_stop_flag) {
        client_pool = new ClientPool;
        Pthread_create(&flood_th, nullptr, flood_loop, nullptr);
    }

    while (g_stop_flag == false) {
        sleep(1);
    }

    if (client_pool != nullptr) {
        delete client_pool;
        Pthread_join(flood_th, nullptr);
    }

    delete server;
    delete svc;
    poll->release();
    thrpool->release();

    RLog::finalize();

    return 0;
}