#include <math.h>

#include "rpc/client.h"
#include "rpc/server.h"
#include "benchmark_service.h"

using namespace benchmark;
using namespace rpc;

static Counter g_nop_counter;

void BenchmarkService::fast_prime(const i32& n, i8* flag) {
    if (n <= 0) {
        *flag = -1;
    } else if (n <= 3) {
        *flag = 1;
    } else if (n % 2 == 0) {
        *flag = 0;
    } else {
        int d = 3;
        int m = sqrt(n) + 1; // +1 for sqrt float errors
        while (d <= m) {
            if (n % d == 0) {
                *flag = 0;
                return;
            }
            d++;
        }
        *flag = 1;
    }
}

void BenchmarkService::nop(const std::string&) {
    int cnt = g_nop_counter.next();
    if (cnt % 10000 == 0) {
        Log::info("%d nop requests", cnt);
    }
}

void BenchmarkService::sleep(const double& sec) {
    int full_sec = (int) sec;
    int usec = int((sec - full_sec) * 1000 * 1000);
    if (full_sec > 0) {
        ::sleep(full_sec);
    }
    if (usec > 0) {
        usleep(usec);
    }
}

void BenchmarkService::add_later(const rpc::i32& a, const rpc::i32& b, rpc::i32* sum, rpc::DeferredReply* defer) {
    *sum = a + b;

    // defer->reply() will cleanup all resource, including `defer` itself
    defer->reply();
}
