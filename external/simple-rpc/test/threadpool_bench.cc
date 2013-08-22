#include <stdio.h>

#include "rpc/utils.h"

using namespace std;
using namespace rpc;

int main(int argc, char* argv[]) {
    const int threadpool_size = 32;

    // benchmark threadpool
    ThreadPool* thrpool = new ThreadPool(threadpool_size);
    Timer timer;
    auto nop_job = [] {};
    int n_jobs = 1000 * 1000;
    timer.start();
    for (int i = 0; i < n_jobs; i++) {
        thrpool->run_async(nop_job);
    }
    thrpool->release();
    timer.end();
    Log_info("running %d nop_jobs in ThreadPool(%d) took %lf seconds, throughput=%lf (1 issuing thread)",
        n_jobs, threadpool_size, timer.elapsed(), n_jobs / timer.elapsed());

}
