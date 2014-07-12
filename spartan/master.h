// Copyright (C) 2014, Peking University
// Author: Qi Chen (chenqi871025@gmail.com)
//
// Description:
#ifndef MASTER_H
#define MASTER_H

#include <unistd.h>
#include <pthread.h>

#include "fastrpc/service.h"
#include "base/logging.h"

class Master: public spartan::MasterService {
private:
    int32_t _num_workers;

public:
    Master() {}
    ~Master() {}

    void reg(const RegisterReq& req) {

    }

    void maybe_steal_tile(const UpdateAndStealTileReq& req, TileIdMessage* resp) {

    }

    void heartbeat(const HeartbeatReq& req) {
        Log_info("receive heartbeat from worker:%d with worker_status:%s", req.worker_id,
                 req.worker_status.to_string().c_str());
    }
};

#endif
