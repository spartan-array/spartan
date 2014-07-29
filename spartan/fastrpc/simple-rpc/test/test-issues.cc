#include <unistd.h>

#include "rpc/server.h"
#include "rpc/client.h"
#include "benchmark_service.h"

using namespace base;
using namespace rpc;
using namespace benchmark;

TEST(issue, 7) {
    const int rpc_id = 1987;
    PollMgr* poll = new PollMgr(1);
    ThreadPool* thrpool = new ThreadPool(4);
    Server *svr = new Server(poll, thrpool);
    // directly release poll to make sure it has ref_count of 1
    // other other hand, thrpool is kept till end of program, with ref_count of 2
    poll->release();
    svr->reg(rpc_id, [] (Request* req, ServerConnection* sconn) {
        sconn->run_async([req, sconn] {
            Log::debug("rpc called");
            ::usleep(500 * 1000);
            Log::debug("rpc replying");
            sconn->begin_reply(req);
            // reply nothing
            sconn->end_reply();
            delete req;
            sconn->release();
            Log::debug("rpc replied");
        });
    });
    svr->start("127.0.0.1:7891");

    PollMgr* poll_clnt = new PollMgr(1);
    Client* clnt = new Client(poll_clnt);
    clnt->connect("127.0.0.1:7891");
    Client* clnt2 = new Client(poll_clnt);
    clnt2->connect("127.0.0.1:7891");
    Client* clnt3 = new Client(poll_clnt);
    clnt3->connect("127.0.0.1:7891");
    Client* clnt4 = new Client(poll_clnt);
    clnt4->connect("127.0.0.1:7891");
    Future* fu = clnt->begin_request(rpc_id);
    clnt->end_request();
    fu->timed_wait(0.1);
    fu->release();
    fu = clnt2->begin_request(rpc_id);
    clnt2->end_request();
    fu->timed_wait(0.1);
    fu->release();
    fu = clnt3->begin_request(rpc_id);
    clnt3->end_request();
    fu->timed_wait(0.1);
    fu->release();
    fu = clnt4->begin_request(rpc_id);
    clnt4->end_request();
    // wait a little bit to make sure RPC got sent instead of cancelled
    fu->timed_wait(0.1);
    fu->release();
    clnt->close_and_release();
    clnt2->close_and_release();
    clnt3->close_and_release();
    clnt4->close_and_release();
    poll_clnt->release();

    Log::debug("killing server");
    delete svr;
    Log::debug("killed server");

    // thrpool is kept till end of program, with ref_count of 2
    thrpool->release();
}


TEST(issue, 8) {
    BenchmarkService svc;
    Server *svr = new Server;
    PollMgr* clnt_poll = new PollMgr(1);
    svr->reg(&svc);
    svr->start("0.0.0.0:9876");
    Client* clnt = new Client(clnt_poll);
    clnt->connect("127.0.0.1:9876");
    BenchmarkProxy bp(clnt);
    Future* fu = bp.async_sleep(2);
    fu->timed_wait(0.1);
    fu->release();
    clnt->close_and_release();
    delete svr;
    clnt_poll->release();
}
