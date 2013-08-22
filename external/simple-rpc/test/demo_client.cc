#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <string>

#include "rpc/client.h"
#include "rpc/server.h"
#include "rpc/marshal.h"

#include "demo_service.h"

#define streq(a, b) (strcmp((a), (b)) == 0)

using namespace std;
using namespace rpc;
using namespace demo;

enum eval_case_t {
    FAST_PRIME,
    FAST_DOT_PROD,
    FAST_LARGE_STR_NOP,
    PRIME,
    DOT_PROD,
    LARGE_STR_NOP
};

enum eval_case_t eval_case;

string long_str(16 * 1024, 'c');

Counter g_rpc_issued;
static int g_n_rpc = 1000 * 1000;

inline point3 rand_pt() {
    point3 pt;
    pt.x = ((double) rand()) / RAND_MAX;
    pt.y = ((double) rand()) / RAND_MAX;
    pt.z = ((double) rand()) / RAND_MAX;
    return pt;
}

inline void do_work(ClientPool* cl_pool, const char* svr_addr, FutureAttr& fu_attr) {
    if (g_rpc_issued.next() > g_n_rpc) {
        return;
    }
    Client* cl = cl_pool->get_client(svr_addr);
    Future* fu;
    i32 prime_test = 1 + rand() % 99;
    switch (eval_case) {
    case FAST_PRIME:
        fu = DemoProxy(cl).async_fast_prime(prime_test, fu_attr);
        break;
    case PRIME:
        fu = DemoProxy(cl).async_prime(prime_test, fu_attr);
        break;
    case FAST_DOT_PROD:
        fu = DemoProxy(cl).async_fast_dot_prod(rand_pt(), rand_pt(), fu_attr);
        break;
    case DOT_PROD:
        fu = DemoProxy(cl).async_dot_prod(rand_pt(), rand_pt(), fu_attr);
        break;
    case FAST_LARGE_STR_NOP:
        fu = DemoProxy(cl).async_fast_large_str_nop(long_str, fu_attr);
        break;
    case LARGE_STR_NOP:
        fu = DemoProxy(cl).async_large_str_nop(long_str, fu_attr);
        break;
    default:
        Log_fatal("unexpected code reached!");
        verify(0);
    }
    if (fu != NULL) {
        fu->release();
    }
}

int main(int argc, char* argv[]) {
    printf("usage: %s svr_addr eval_case\n", argv[0]);
    printf("eval_case: fast_prime, fast_dot_prod, fast_large_str_nop, prime, dot_prod, large_str_nop\n");
    if (argc < 3) {
        exit(1);
    }

    char* svr_addr = argv[1];
    if (streq(argv[2], "fast_prime")) {
        eval_case = FAST_PRIME;
    } else if (streq(argv[2], "fast_dot_prod")) {
        eval_case = FAST_DOT_PROD;
    } else if (streq(argv[2], "fast_large_str_nop")) {
        eval_case = FAST_LARGE_STR_NOP;
    } else if (streq(argv[2], "prime")) {
        eval_case = PRIME;
    } else if (streq(argv[2], "dot_prod")) {
        eval_case = DOT_PROD;
    } else if (streq(argv[2], "large_str_nop")) {
        eval_case = LARGE_STR_NOP;
    } else {
        Log_fatal("eval case not supported: %s", argv[2]);
        exit(1);
    }

    srand(getpid());
    PollMgr* poll = new PollMgr;
    ClientPool* cl_pool = new ClientPool(poll, 1);

    const int concurrency = 1;
    Counter rpc_counter;

    FutureAttr attr1;
    FutureAttr attr2;
    verify(attr1.callback == nullptr);
    attr1.callback = [&] (Future*) { do_work(cl_pool, svr_addr, attr2); rpc_counter.next(); };
    attr2.callback = [&] (Future*) { do_work(cl_pool, svr_addr, attr1); rpc_counter.next(); };

    for (int i = 0; i < concurrency; i++) {
        do_work(cl_pool, svr_addr, attr1);
    }

    for (int i = 0; i < 20; i++) {
        Log_debug("clock tick, about %d rpc done", rpc_counter.peek_next());
        if (g_rpc_issued.peek_next() > g_n_rpc) {
            break;
        }
        sleep(1);
    }

    delete cl_pool;
    poll->release();
    return 0;
}
