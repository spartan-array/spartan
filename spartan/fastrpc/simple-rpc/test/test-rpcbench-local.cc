#include "base/all.h"
#include "rpc/client.h"
#include "rpc/server.h"
#include "benchmark_service.h"

using namespace base;
using namespace std;
using namespace rpc;
using namespace benchmark;

TEST(integration, rpc_bench_local) {
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
    Timer timer;

    const int n_prime = 100000;
    FutureGroup* fu_group = new FutureGroup;

    timer.start();
    for (int i = 0; i < n_prime; i++) {
        i8 flag = 0;
        clnt->fast_prime(i + 1987, &flag);
    }
    timer.stop();
    Log::debug("fast_prime, sync: %.0lf", n_prime / timer.elapsed());

    timer.start();
    for (int i = 0; i < n_prime; i++) {
        fu_group->add(clnt->async_fast_prime(i + 1987));
    }
    delete fu_group;
    timer.stop();
    Log::debug("fast_prime, async: %.0lf", n_prime / timer.elapsed());


    timer.start();
    for (int i = 0; i < n_prime; i++) {
        i8 flag = 0;
        clnt->prime(i + 1987, &flag);
    }
    timer.stop();
    Log::debug("prime, sync: %.0lf", n_prime / timer.elapsed());

    fu_group = new FutureGroup;
    timer.start();
    for (int i = 0; i < n_prime; i++) {
        fu_group->add(clnt->async_prime(i + 1987));
    }
    delete fu_group;
    timer.stop();
    Log::debug("prime, async: %.0lf", n_prime / timer.elapsed());


    const int n_dot_prod = 100000;
    timer.start();
    for (int i = 0; i < n_dot_prod; i++) {
        point3 a;
        a.x = (double) i;
        a.y = (double) i;
        a.z = (double) i;
        point3 b;
        b.x = (double) i - 1987;
        b.y = (double) i - 1987;
        b.z = (double) i - 1987;
        double v;
        clnt->fast_dot_prod(a, b, &v);
    }
    timer.stop();
    Log::debug("fast_dot_prod, sync: %.0lf", n_dot_prod / timer.elapsed());

    fu_group = new FutureGroup;
    timer.start();
    for (int i = 0; i < n_dot_prod; i++) {
        point3 a;
        a.x = (double) i;
        a.y = (double) i;
        a.z = (double) i;
        point3 b;
        b.x = (double) i - 1987;
        b.y = (double) i - 1987;
        b.z = (double) i - 1987;
        fu_group->add(clnt->async_fast_dot_prod(a, b));
    }
    delete fu_group;
    timer.stop();
    Log::debug("fast_dot_prod, async: %.0lf", n_dot_prod / timer.elapsed());


    timer.start();
    for (int i = 0; i < n_dot_prod; i++) {
        point3 a;
        a.x = (double) i;
        a.y = (double) i;
        a.z = (double) i;
        point3 b;
        b.x = (double) i - 1987;
        b.y = (double) i - 1987;
        b.z = (double) i - 1987;
        double v;
        clnt->dot_prod(a, b, &v);
    }
    timer.stop();
    Log::debug("dot_prod, sync: %.0lf", n_dot_prod / timer.elapsed());

    fu_group = new FutureGroup;
    timer.start();
    for (int i = 0; i < n_dot_prod; i++) {
        point3 a;
        a.x = (double) i;
        a.y = (double) i;
        a.z = (double) i;
        point3 b;
        b.x = (double) i - 1987;
        b.y = (double) i - 1987;
        b.z = (double) i - 1987;
        fu_group->add(clnt->async_dot_prod(a, b));
    }
    delete fu_group;
    timer.stop();
    Log::debug("dot_prod, async: %.0lf", n_dot_prod / timer.elapsed());


    const int n_add = 100000;
    timer.start();
    for (int i = 0; i < n_add; i++) {
        v32 i_add_1987;
        clnt->fast_add(i, 1987, &i_add_1987);
    }
    timer.stop();
    Log::debug("fast_add, sync: %.0lf", n_add / timer.elapsed());

    fu_group = new FutureGroup;
    timer.start();
    for (int i = 0; i < n_add; i++) {
        fu_group->add(clnt->async_fast_add(i, 1987));
    }
    delete fu_group;
    timer.stop();
    Log::debug("fast_add, async: %.0lf", n_add / timer.elapsed());


    timer.start();
    for (int i = 0; i < n_add; i++) {
        v32 i_add_1987;
        clnt->add(i, 1987, &i_add_1987);
    }
    timer.stop();
    Log::debug("add, sync: %.0lf", n_add / timer.elapsed());

    fu_group = new FutureGroup;
    timer.start();
    for (int i = 0; i < n_add; i++) {
        fu_group->add(clnt->async_add(i, 1987));
    }
    delete fu_group;
    timer.stop();
    Log::debug("add, async: %.0lf", n_add / timer.elapsed());

    delete clnt;
    delete clnt_pool;
    delete svr;

    thrpool->release();
    poll->release();
}
