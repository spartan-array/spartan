#include <stdio.h>
#include <stdlib.h>
#include <semaphore.h>
#include <unistd.h>
#include <sys/time.h>
#include <signal.h>

#include "rpc/client.h"
#include "rpc/server.h"
#include "demo_service.h"

#define NUM 10000000

using namespace demo;
using namespace rpc;

int n_th = 1;
int n_batch = 1;

typedef struct {
    NullProxy *np;
    int *counter;
    int n_outstanding;
    sem_t sem;
} clt_data;

void client_cb(clt_data* cl, Future* fu) {
    __sync_add_and_fetch(&cl->n_outstanding, -1);
    if (cl->n_outstanding < (n_batch/2)) {
        verify(sem_post(&cl->sem)==0);
    }
}


int
diff_timespec(const struct timespec &end, const struct timespec &start)
{
    int diff = (end.tv_sec > start.tv_sec)?(end.tv_sec-start.tv_sec)*1000:0;
    verify(diff || end.tv_sec == start.tv_sec);
    if (end.tv_nsec > start.tv_nsec) {
        diff += (end.tv_nsec-start.tv_nsec)/1000000;
    } else {
        diff -= (start.tv_nsec-end.tv_nsec)/1000000;
    }
    return diff;
}

void NullService::test(const i32& arg1, const i32& arg2, i32* result) {
    *result = arg1 ^ arg2;
}

rpc::Rand g_rand;

void *
clt_run(void *x)
{
    clt_data *d = (clt_data *)x;
    if (n_batch == 1) {
        for (int i = 0; i < NUM; i++) {
            i32 x,y,r;
            x = g_rand();
            y = g_rand();
            d->np->test(x,y,&r);
            verify(r == (x ^ y));
            *d->counter = *d->counter + 1;
        }
        printf("client finished\n");
    } else {
        FutureAttr attr;
        attr.callback = std::bind(client_cb, d, std::placeholders::_1);

        i32 x,y;
        d->n_outstanding = 0;

        while (1) {
            verify(sem_wait(&d->sem)==0);
            int diff = n_batch - d->n_outstanding;
            if (diff > n_batch/2) {
                for (int i = 0; i < diff; i++) {
                    Future *fu = d->np->async_test(x,y, attr);
                    if (fu) {
                        fu->release();
                    }
                }
                *d->counter = *d->counter + diff;
                /*don't get out of the loop, i'll live with that*/
                if (*d->counter == NUM) sleep(1);

                __sync_add_and_fetch(&d->n_outstanding, diff);
            }
        }
    }

    return NULL;
}

void *
print_stat(void *x)
{
    int *allcounters = (int *)x;
    int last = 0, curr = 0;
    struct timeval now, past;
    gettimeofday(&now, 0);
    do {
        last = curr;
        past = now;
        curr = 0;
        for (int i = 0; i < n_th; i++) {
            curr += allcounters[i];
        }
        gettimeofday(&now, 0);
        double diff_sec = now.tv_sec - past.tv_sec + (now.tv_usec - past.tv_usec) / 1000000.0;
        printf("%.2f s processed %d rpcs = %.2f rpcs/sec\n", diff_sec, curr-last, (curr-last)/diff_sec);
        sleep(1);
    }while (!curr || curr != last);
    return NULL;
}

int main(int argc, char **argv) {
    signal(SIGPIPE, SIG_IGN);
    signal(SIGHUP, SIG_IGN);
    signal(SIGCHLD, SIG_IGN);

    bool isclient = false, isserver = false;
    int num_clients = 0;
    const char *svr_addr = "127.0.0.1:7777";

    if (argc < 2) {
        printf("usage: perftest -s|-c ip:port  -t <num_client_threads> -b <batch_size>\n");
        exit(1);
    }

    char ch = 0;
    while ((ch = getopt(argc, argv, "s:c:t:b:"))!= -1) {
        switch (ch) {
        case 'c':
            isclient = true;
            if (optarg) svr_addr = optarg;
            break;
        case 's':
            isserver = true;
            if (optarg) svr_addr = optarg;
            break;
        case 't':
            n_th = atoi(optarg);
            break;
        case 'b': /* batch of simultaneous rpcs */
            n_batch = atoi(optarg);
            break;
        default:
            break;
        }
    }

    verify(isserver || isclient);

    const int n_io_threads = 8;
    const int n_worker_threads = 64;

    poll_options poll_opts;
    poll_opts.n_threads = n_io_threads;
    PollMgr* poll = new PollMgr(poll_opts);

    if (isserver) {
        printf("starting server on %s\n", svr_addr);
        ThreadPool* thrpool = new ThreadPool(n_worker_threads);
        Server svr(poll, thrpool);
        thrpool->release();
        poll->release();

        NullService null_svc;
        svr.reg(&null_svc);
        svr.start(svr_addr);

        for (;;) {
            sleep(1);
        }
        exit(0);
    } else { //isclient

        if (!num_clients)
            num_clients = n_th;

        NullProxy** allclients = (NullProxy **)malloc(sizeof(NullProxy *)*num_clients);

        pthread_t *cltth = (pthread_t *)malloc(sizeof(pthread_t)*n_th);
        int * counters = (int *)malloc(sizeof(int)*n_th);
        bzero(counters, sizeof(int)*n_th);
        printf("Perf client to create %d rpc clients and %d threads\n", num_clients, n_th);

        for (int i = 0; i < num_clients; i++) {
            Client *cl = new Client(poll);
            verify(cl->connect(svr_addr) == 0);
            allclients[i] = new NullProxy(cl);
        }

        clt_data args[n_th];
        for (int i = 0; i < n_th; i++) {
            args[i].np = allclients[i % num_clients];
            args[i].counter = &counters[i];
            args[i].n_outstanding = 0;
            verify(sem_init(&args[i].sem, 0, 1)==0);
            Pthread_create(&cltth[i], NULL, clt_run, (void *)&args[i]);
        }

        pthread_t stat_th;
        Pthread_create(&stat_th,NULL, print_stat, (void *)counters);

        for (int i = 0; i < n_th; i++) {
            pthread_join(cltth[i], NULL);
        }
    }
}
