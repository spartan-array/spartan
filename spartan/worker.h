// Copyright (C) 2014, Peking University
// Author: Qi Chen (chenqi871025@gmail.com)
//
// Description:
#ifndef WORKER_H
#define WORKER_H

#include <ctime>
#include <unordered_map>
#include <errno.h>
#include <unistd.h>
#include <pthread.h>

#include "array/tile.h"
#include "base/logging.h"
#include "base/threading.h"
#include "fastrpc/service.h"

class BlobCTX;

class Worker: public spartan::WorkerService {
private:
    int32_t id;
    std::string _addr;
    bool _initialized, _running;
    spartan::MasterProxy* _master;
    std::unordered_map<int32_t, spartan::WorkerProxy*> _peers;
    std::unordered_map<TileId, Tile> _blobs;
    std::vector<TileId> _kernel_remain_tiles;
    BlobCTX* _ctx;
    WorkerStatus* _worker_status;
    //pthread_mutex_t _lock;
    base::SpinLock _lock;

    rpc::PollMgr* _clt_poll;
    rpc::ClientPool* _clt_pool;

    int32_t HEARTBEAT_INTERVAL = 3;
    int32_t RPC_TIMEOUT = 1;

    inline void lock() {
        //Pthread_mutex_lock(&_lock);
        _lock.lock();
    }

    inline void unlock() {
        //Pthread_mutex_unlock(&_lock);
        _lock.unlock();
    }

public:
    Worker(const std::string& master_addr, const std::string& worker_addr);
    ~Worker();
    void register_to_master();
    void wait_for_shutdown();
    inline void shutdown() {
        Log_info("Closing server %d ...", id);
        _running = false;
    }

// all the rpc services
public:
    void initialize(const InitializeReq& req, EmptyMessage* resp);
    void get_tile_info(const TileIdMessage& req, RunKernelResp* resp);
    void create(const CreateTileReq& req, TileIdMessage* resp);
    void destroy(const DestroyReq& req, EmptyMessage* resp);
    void update(const UpdateReq& req, EmptyMessage* resp);
    void get(const GetReq& req, GetResp* resp);
    void get_flatten(const GetReq& req, GetResp* resp);
    void cancel_tile(const TileIdMessage& req, rpc::i8* resp);
    void run_kernel(const RunKernelReq& req, RunKernelResp* resp);
    void shutdown(const EmptyMessage& req, EmptyMessage* resp) { shutdown(); }
};

#endif
