// Copyright (C) 2014, Peking University
// Author: Qi Chen (chenqi871025@gmail.com)
//
// Description:
//
#ifndef BLOBCTX_H
#define BLOBCTX_H

#include "base/logging.h"
#include "core.h"
#include "rpc/client.h"
#include "fastrpc/service.h"
#include "worker.h"

class BlobCTX {
private:
    int32_t worker_id;
    Worker* local_worker;
    int32_t num_workers;
    std::unordered_map<int32_t, spartan::WorkerProxy*>* workers;
    static int32_t RPC_TIMEOUT;

    template<typename Q, typename R>
    rpc::Future* _send(int32_t worker_id,
                       void (Worker::*pLocalFunc)(const Q&, R*),
                       rpc::Future* (spartan::WorkerProxy::*pFunc)(const Q&, const rpc::FutureAttr&),
                       Q& req, R& resp, bool wait=true, int32_t timeout=BlobCTX::RPC_TIMEOUT) const {

        rpc::Future* fu = NULL;
        if (worker_id == this->worker_id) {
            (local_worker->*pLocalFunc)(req, &resp);
        } else {
            std::unordered_map<int32_t, spartan::WorkerProxy*>::iterator it = workers->find(worker_id);

            if (it == workers->end()) {
                Log_error("Cannot find worker_id:%d", worker_id);
                return NULL;
            }

            rpc::FutureAttr fu_attr;
            fu_attr.callback = [&resp] (rpc::Future* fu) {
                if (fu->get_error_code() == 0) {
                    fu->get_reply() >> resp;
                } else {
                    Log_error("Receive error code:%d", fu->get_error_code());
                }
            };

            fu = ((it->second)->*pFunc)(req, fu_attr);
            if (fu == NULL) {
                Log_error("cannot connect to worker:%d", worker_id);
                return NULL;
            }

            if (!wait) return fu;

            fu->timed_wait(timeout);
            fu->release();
        }

        return NULL;
    }

    template<typename Q, typename R>
    void _send_all(rpc::Future* (spartan::WorkerProxy::*pFunc)(const Q&, const rpc::FutureAttr&),
                               Q& req, std::vector<R*>* resp,
                               std::vector<int32_t>* target_workers=NULL,
                               bool wait=true, int32_t timeout=BlobCTX::RPC_TIMEOUT) const {
        rpc::Future* fu = NULL;
        std::vector<rpc::Future*> fu_group;

        rpc::FutureAttr fu_attr;
        if (resp != NULL) {
            fu_attr.callback = [resp] (rpc::Future* fu) {
                if (fu->get_error_code() == 0) {
                    R* re = new R();
                    fu->get_reply() >> *re;
                    resp->push_back(re);
                } else {
                    Log_error("Receive error code:%d", fu->get_error_code());
                }
            };
        } else {
            fu_attr.callback = [] (rpc::Future* fu) {
                if (fu->get_error_code() != 0) {
                    Log_error("Receive error code:%d", fu->get_error_code());
                }
                fu->release();
            };
        }

        if (target_workers == NULL) {
            for (const auto& it : *workers) {
                fu = ((it.second)->*pFunc)(req, fu_attr);
                if (fu == NULL) {
                    Log_error("cannot connect to worker:%d", it.first);
                    continue;
                }

                fu_group.push_back(fu);
            }
        } else {
            for (const auto& w : *target_workers) {
                std::unordered_map<int32_t, spartan::WorkerProxy*>::iterator it = workers->find(w);

                if (it == workers->end()) {
                    Log_error("Cannot find worker_id:%d", w);
                    continue;
                }

                fu = ((it->second)->*pFunc)(req, fu_attr);
                if (fu == NULL) {
                    Log_error("cannot connect to worker:%d", w);
                    continue;
                }

                fu_group.push_back(fu);
            }
        }

        if (!wait) return;

        for (auto& f : fu_group) {
            f->timed_wait(timeout);
            f->release();
        }
    }

public:
    BlobCTX(int32_t wid, std::unordered_map<int32_t, spartan::WorkerProxy*>* peers, Worker* l_worker = NULL):
        worker_id(wid), local_worker(l_worker), num_workers(peers->size()), workers(peers) {
        RPC_TIMEOUT = 1000;
    }

    ~BlobCTX() {}

    bool is_master() const {
        return local_worker == NULL;
    }

    void destroy_all(std::vector<TileId>& tile_ids) const {
        DestroyReq req(tile_ids);
        _send_all<DestroyReq, EmptyMessage>(&spartan::WorkerProxy::async_destroy, req,
                                            NULL, NULL, false);
    }

    void destroy(const TileId& tile_id) const {
        std::vector<TileId> tids;
        tids.push_back(tile_id);
        destroy_all(tids);
    }

    rpc::Future* get(const TileId& tile_id, const SubSlice& subslice, GetResp& resp,
                     bool wait=true, int32_t timeout=BlobCTX::RPC_TIMEOUT) {
        GetReq req(tile_id, subslice);
        return _send(tile_id.worker, &Worker::get, &spartan::WorkerProxy::async_get,
                     req, resp, wait, timeout);
    }

    void map(std::vector<TileId>& tile_ids, std::string& mapper_fn) {
        std::vector<RunKernelResp*> resp;
        RunKernelReq req(tile_ids, mapper_fn);
        _send_all<RunKernelReq, RunKernelResp>(&spartan::WorkerProxy::async_run_kernel,
                                               req, &resp);

        for (auto& re : resp) {
            Log_debug("receive re:%s", (re->result).c_str());
        }
    }
};

int32_t BlobCTX::RPC_TIMEOUT = 1000;

#endif

