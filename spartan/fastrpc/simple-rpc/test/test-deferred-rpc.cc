#include "base/all.h"

#include "rpc/server.h"
#include "rpc/client.h"
#include "benchmark_service.h"

using namespace std;
using namespace rpc;
using namespace benchmark;

TEST(rpc, deferred_reply) {
    Server* svr = new Server;
    BenchmarkService bench_svc;
    svr->reg(&bench_svc);
    const char* svr_addr = "127.0.0.1:1987";
    svr->start(svr_addr);

    PollMgr* clnt_poll = new PollMgr;
    ClientPool* clnt_pool = new ClientPool(clnt_poll);
    BenchmarkProxy* clnt = new BenchmarkProxy(clnt_pool->get_client(svr_addr));

    i32 a = 7;
    i32 b = 8;
    i32 r = 0;
    clnt->add_later(a, b, &r);
    EXPECT_EQ(r, a + b);

    delete clnt;
    delete clnt_pool;

    delete svr;
    clnt_poll->release();
}
