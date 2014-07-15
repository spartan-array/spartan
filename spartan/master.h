// Copyright (C) 2014, Peking University
// Author: Qi Chen (chenqi871025@gmail.com)
//
// Description:
#ifndef MASTER_H
#define MASTER_H

#include <unistd.h>
#include <pthread.h>

#include "core.h"
#include "base/logging.h"
#include "blobctx.h"
#include "fastrpc/service.h"

#define MASTER_ID 65535

class Master: public spartan::MasterService {
private:
    int32_t _num_workers;
    std::unordered_map<int32_t, std::string> _workers;

    rpc::PollMgr* _clt_poll;
    rpc::ClientPool* _clt_pool;
    std::unordered_map<int32_t, spartan::WorkerProxy*> _available_workers;

    bool _initialized, _running;
    BlobCTX* _ctx;

    pthread_t tid;
    std::unordered_map<int32_t, WorkerStatus> _worker_status;
    std::unordered_map<int32_t, double> _worker_score;
    //std::set<DistArray> _arrays;

    static void* _initialize(void* args);

public:
    Master(int32_t nw) {
        _num_workers = nw;
        _initialized = false;
        _running = true;
        _ctx = NULL;
        _clt_poll = new rpc::PollMgr;
        _clt_pool = new rpc::ClientPool(_clt_poll);
    }

    ~Master() {
        delete _ctx;
        for (auto& it : _available_workers) {
            delete it.second;
        }
        _available_workers.clear();

        delete _clt_pool;
        _clt_poll->release();
    }

    void reg(const RegisterReq& req, EmptyMessage* resp) {
        int32_t id = _workers.size();
        _workers[id] = req.host;
        _available_workers[id] = new spartan::WorkerProxy(_clt_pool->get_client(req.host));
        Log_info("Registered %s (%d/%d)", req.host.c_str(), id, _num_workers);

        if ((int32_t)_workers.size() >= _num_workers) {
            Pthread_create(&tid, NULL, _initialize, this);
        }
    }

    void maybe_steal_tile(const UpdateAndStealTileReq& req, TileIdMessage* resp) {

    }

    void heartbeat(const HeartbeatReq& req, EmptyMessage* resp) {
        Log_info("receive heartbeat from worker:%d with worker_status:%s", req.worker_id,
                 req.worker_status.to_string().c_str());
    }
};

void* Master::_initialize(void* args) {
    Master* pThis = (Master*)args;

    Log_info("Initializing...");
    InitializeReq req(-1, pThis->_workers);
    rpc::FutureGroup fu_group;
    rpc::FutureAttr fu_attr;
    for (auto& it : pThis->_available_workers) {
        req.id = it.first;
        fu_group.add((it.second)->async_initialize(req, fu_attr));
    }
    fu_group.wait_all();

    pThis->_ctx = new BlobCTX(MASTER_ID, &(pThis->_available_workers), NULL);

    pThis->_initialized = true;
    Log_info("Init master done...");

    TileId tid(0,1);
    std::vector<TileId> tids;
    tids.push_back(tid);
    std::string mapper_fn = "add";
    pThis->_ctx->map(tids, mapper_fn);
    pThis->_ctx->destroy_all(tids);
    return NULL;
}


#endif
