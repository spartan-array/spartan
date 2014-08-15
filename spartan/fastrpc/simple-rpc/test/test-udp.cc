#include "base/all.h"
#include "rpc/client.h"
#include "rpc/server.h"
#include "benchmark_service.h"

using namespace base;
using namespace std;
using namespace rpc;
using namespace benchmark;


template <class T>
void report_qps(const char* action, T n_ops, double duration) {
    base::Log::info("%s: %d ops, took %.2lf sec, qps=%s",
        action, n_ops, duration, base::format_decimal(T(n_ops / duration)).c_str());
}

TEST(udp, server_start_stop) {
    Server* svr = new Server;
    BenchmarkService bench_svc;
    svr->reg(&bench_svc);
    svr->start("localhost:8848");
    delete svr;
}


TEST(udp, simple_rpc) {
    Server* svr = new Server;
    BenchmarkService bench_svc;
    svr->reg(&bench_svc);
    svr->start("localhost:8848");
    PollMgr* poll = new PollMgr;
    Client* clnt = new Client(poll);
    clnt->connect("localhost:8848");
    BenchmarkProxy proxy(clnt);

    i32 p = 17;
    i8 flag = 0;
    proxy.fast_prime(p, &flag);
    Log::debug("prime(%d) -> %d", p, flag);

    const int batch_size = 10 * 1000;
    int n_batches = 0;
    Timer timer;
    timer.start();
    for (;;) {
        for (int i = 0; i < batch_size; i++) {
            i32 dummy1 = 1987;
            i32 dummy2 = 1989;
            proxy.lossy_nop(dummy1, dummy2);
//            proxy.fast_lossy_nop();
        }
        n_batches++;
        if (timer.elapsed() > 2.0) {
            break;
        }
    }
    timer.stop();
    int n_rpc = n_batches * batch_size;
    report_qps("client UDP RPCs", n_rpc, timer.elapsed());

    clnt->close_and_release();
    poll->release();
    delete svr;
}
