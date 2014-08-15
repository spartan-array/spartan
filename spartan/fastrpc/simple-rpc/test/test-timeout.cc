#include "base/all.h"
#include "rpc/client.h"
#include "rpc/server.h"
#include "benchmark_service.h"

using namespace base;
using namespace std;
using namespace rpc;
using namespace benchmark;

TEST(future, wait_timeout) {
    PollMgr* poll = new PollMgr;
    ThreadPool* thrpool = new ThreadPool;

    // start the server
    int svc_port = 1987;
    EXPECT_NEQ(svc_port, -1);
    Server* svr = new Server(poll, thrpool);
    BenchmarkService bench_svc;
    svr->reg(&bench_svc);
    char svc_addr[100];
    sprintf(svc_addr, "127.0.0.1:%d", svc_port);
    svr->start(svc_addr);

    // start the client
    ClientPool* clnt_pool = new ClientPool(poll);
    BenchmarkProxy* clnt = new BenchmarkProxy(clnt_pool->get_client(svc_addr));

    Log::debug("do wait");
    Timer t;
    t.start();
    FutureAttr fu_attr;
    fu_attr.callback = [] (Future* fu) {
        Log::debug("fu->get_error_code() = %d", fu->get_error_code());
    };
    Future* fu = clnt->async_sleep(2.3, fu_attr);
    double wait_sec = 1.0;
    fu->timed_wait(wait_sec);
    fu->release();
    t.stop();
    Log::debug("done wait: %lf seconds", t.elapsed());
    EXPECT_LT(fabs(wait_sec - t.elapsed()), 0.1);

    delete clnt;
    delete clnt_pool;
    delete svr;

    thrpool->release();
    poll->release();
}
