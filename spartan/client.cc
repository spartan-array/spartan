#include "base/all.h"

#include "rpc/server.h"
#include "rpc/client.h"
#include "fastrpc/service.h"
#include "core.h"

using namespace std;
using namespace rpc;
using namespace spartan;

int main() {
    PollMgr* clnt_poll = new PollMgr;
    ClientPool* clnt_pool = new ClientPool(clnt_poll);
    WorkerProxy* clnt = new WorkerProxy(clnt_pool->get_client("0.0.0.0:1111"));

    RunKernelResp r;
    TileId tid(0, 1);
    TileIdMessage m(tid);
    clnt->get_tile_info(m, &r);

    delete clnt;
    delete clnt_pool;
    clnt_poll->release();
    return 0;
}
