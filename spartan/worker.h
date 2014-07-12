// Copyright (C) 2014, Peking University
// Author: Qi Chen (chenqi871025@gmail.com)
//
// Description:
#ifndef WORKER_H
#define WORKER_H

#include <ctime>
#include <errno.h>
#include <unistd.h>
#include <pthread.h>

#include "array/tile.h"
#include "fastrpc/service.h"
#include "base/logging.h"

class Worker: public spartan::WorkerService {
private:
    int32_t id;
    bool _initialized, _running;
    std::map<int32_t, spartan::WorkerProxy> _peers;
    spartan::MasterProxy* _master;
    std::map<TileId, Tile> _blobs;
    std::vector<TileId> _kernel_remain_tiles;
    //BlobCTX _ctx;
    WorkerStatus* _worker_status;
    pthread_mutex_t _lock;

    rpc::PollMgr* _clt_poll;
    rpc::ClientPool* _clt_pool;

    int32_t HEARTBEAT_INTERVAL;

public:
    Worker() {
        id = -1;
        _initialized = true;
        _running = true;
        _clt_poll = new rpc::PollMgr;
        _clt_pool = new rpc::ClientPool(_clt_poll);
        _master = new spartan::MasterProxy(_clt_pool->get_client("0.0.0.0:1112"));
        _worker_status = new WorkerStatus(sysconf(_SC_PHYS_PAGES) * sysconf(_SC_PAGE_SIZE),
                                          (int32_t)sysconf(_SC_NPROCESSORS_CONF),
                                          0, 0, (double)time(0), _kernel_remain_tiles);
        Pthread_mutex_init(&_lock, NULL);
        HEARTBEAT_INTERVAL = 3;
    }

    ~Worker() {
        delete _master;
        delete _worker_status;
        Pthread_mutex_destroy(&_lock);

        delete _clt_pool;
        _clt_poll->release();
    }

    void wait_for_shutdown() {
        double last_heartbeat = (double)time(0);
        while (_running) {
            double now = (double)time(0);
            if (now - last_heartbeat < HEARTBEAT_INTERVAL || !_initialized) {
                sleep(0.1);
                continue;
            }

            _worker_status->update_status(0, 0, now);
            HeartbeatReq req(id, *_worker_status);

            rpc::FutureAttr fu_attr;
            fu_attr.callback = [] (rpc::Future* fu) {
                if (fu->get_error_code() == ETIMEDOUT) {
                    Log_error("Exit due to heartbeat message timeout.");
                    exit(0);
                }
            };
            rpc::Future* fu = _master->async_heartbeat(req, fu_attr);
            fu->timed_wait(1.0);
            fu->release();

            last_heartbeat = (double)time(0);
        }
        Log_info("Worker %d shutdown. Exiting.", id);
    }

// all the rpc services
public:
    void initialize(const InitializeReq& req) {

    }

    void get_tile_info(const TileIdMessage& req, RunKernelResp* resp) {
        Log_info("receive get_tile_info %s", req.tile_id.to_string().c_str());
        Log_info("worker status %s", _worker_status->to_string().c_str());
    }

    void create(const CreateTileReq& req, TileIdMessage* resp) {

    }

    void destroy(const DestroyReq& req) {

    }

    void update(const UpdateReq& req) {

    }

    void get(const GetReq& req, GetResp* resp) {

    }

    void get_flatten(const GetReq& req, GetResp* resp) {

    }

    void cancel_tile(const TileIdMessage& req, rpc::i8* resp) {

    }

    void run_kernel(const RunKernelReq& req, RunKernelResp* resp) {

    }

    void shutdown() {
        Log_info("Closing server %d ...", id);
        _running = false;
    }
};

#endif
