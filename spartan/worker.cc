// Copyright (C) 2014, Peking University
// Author: Qi Chen (chenqi871025@gmail.com)
//
// Description:
//
//
#include <string>
#include <stdio.h>
#include <signal.h>

#include "blobctx.h"
#include "worker.h"

Worker::Worker(const std::string& master_addr, const std::string& worker_addr) {
    id = -1;
    _addr = worker_addr;
    _initialized = false;
    _running = true;
    _ctx = NULL;
    _clt_poll = new rpc::PollMgr;
    _clt_pool = new rpc::ClientPool(_clt_poll);
    _master = new spartan::MasterProxy(_clt_pool->get_client(master_addr));
    _worker_status = new WorkerStatus(sysconf(_SC_PHYS_PAGES) * sysconf(_SC_PAGE_SIZE),
                                      (int32_t)sysconf(_SC_NPROCESSORS_CONF),
                                      0, 0, (double)time(0), _kernel_remain_tiles);
    //Pthread_mutex_init(&_lock, NULL);
}

void Worker::register_to_master() {
    RegisterReq req(_addr, *_worker_status);
    rpc::FutureAttr fu_attr;
    fu_attr.callback = [] (rpc::Future* fu) {
        if (fu->get_error_code() != 0) {
            Log_error("Exit due to register message error:%d.", fu->get_error_code());
            exit(0);
        }
        fu->release();
    };
    rpc::Future* fu = _master->async_reg(req, fu_attr);
    if (fu == NULL) {
        Log_error("Exit due to connection to master failed.");
        exit(0);
    }
}

Worker::~Worker() {
    delete _ctx;
    for (auto& it : _peers) {
        delete it.second;
    }
    _peers.clear();

    delete _master;
    delete _worker_status;
    //Pthread_mutex_destroy(&_lock);

    delete _clt_pool;
    _clt_poll->release();
}
void Worker::wait_for_shutdown() {
    double last_heartbeat = (double)time(0);
    while (_running) {
        double now = (double)time(0);
        if (now - last_heartbeat < HEARTBEAT_INTERVAL || !_initialized) {
            sleep(0.5);
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
        if (fu == NULL) {
            Log_error("Exit due to connection to master failed.");
            exit(0);
        }
        fu->timed_wait(1);
        fu->release();

        last_heartbeat = (double)time(0);
    }
    Log_info("Worker %d shutdown. Exiting.", id);
}

/**
 * Initialize worker.
 * Assigns this worker a unique identifier and sets up connections to all other workers in the process.
 *
 * Args:
 *     req (InitializeReq): foo
 *     resp (EmptyMessage): bar
 */

void Worker::initialize(const InitializeReq& req, EmptyMessage* resp) {
    Log_debug("Worker %d initializing...", req.id);
    for (auto& it : req.peers) {
        _peers[it.first] = new spartan::WorkerProxy(_clt_pool->get_client(it.second));
    }
    id = req.id;
    _ctx = new BlobCTX(id, &_peers, this);
    // set global ctx
    _initialized = true;
}

void Worker::get_tile_info(const TileIdMessage& req, RunKernelResp* resp) {
    Log_info("receive get_tile_info %s", req.tile_id.to_string().c_str());
    Log_info("worker status %s", _worker_status->to_string().c_str());
}

void Worker::create(const CreateTileReq& req, TileIdMessage* resp) {

}

void Worker::destroy(const DestroyReq& req, EmptyMessage* resp) {
    Log_debug("get destroy req");
    for (auto& tid : req.ids) {
        Log_debug("destroy tile:%s", tid.to_string().c_str());
    }
}

void Worker::update(const UpdateReq& req, EmptyMessage* resp) {

}

void Worker::get(const GetReq& req, GetResp* resp) {
    Log_debug("receive get %s[%d:%d:%d]", req.id.to_string().c_str(),
              req.subslice.slices[0].start, req.subslice.slices[0].stop, req.subslice.slices[0].step);
    resp->data = "get success!";
}

void Worker::get_flatten(const GetReq& req, GetResp* resp) {

}

void Worker::cancel_tile(const TileIdMessage& req, rpc::i8* resp) {

}

void Worker::run_kernel(const RunKernelReq& req, RunKernelResp* resp) {
    Log_debug("receive run_kernel req");
    for (auto& tid : req.blobs) {
        Log_debug("tile:%s", tid.to_string().c_str());
    }
    Log_debug("fn:%s", req.fn.c_str());
    resp->result = "run_kernel success in worker " + std::to_string(id);
}

Worker * w;

static void signal_handler(int sig) {
    Log_info("caught signal %d, stopping server now", sig);
    w->shutdown();
}

int main(int argc, char* argv[]) {
    std::string master_addr = "0.0.0.0:1112";
    std::string worker_addr = "0.0.0.0:1111";
    if (argc >= 2) {
        worker_addr = argv[1];
    }

    w = new Worker(master_addr, worker_addr);
    rpc::Server *server = new rpc::Server();
    server->reg(w);

    int ret;
    if ((ret = server->start(worker_addr.c_str())) == 0) {
        signal(SIGPIPE, SIG_IGN);
        signal(SIGHUP, SIG_IGN);
        signal(SIGCHLD, SIG_IGN);

        signal(SIGALRM, signal_handler);
        signal(SIGINT, signal_handler);
        signal(SIGQUIT, signal_handler);
        signal(SIGTERM, signal_handler);

        w->register_to_master();
        w->wait_for_shutdown();
    }

    delete server;
    delete w;

    return ret;
}

