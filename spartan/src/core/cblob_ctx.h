#ifndef CBLOBCTX_H
#define CBLOBCTX_H

#include "base/logging.h"
#include "ccore.h"
#include "cworker.h"
#include "fastrpc/service.h"
#include "rpc/client.h"

class CBlobCtx {
private:
    int32_t worker_id; // Identifier for this worker
    CWorker* local_worker; // A reference to the local worker creating this context.
    int32_t num_workers;
    std::unordered_map<int32_t, spartan::WorkerProxy*>* workers; // RPC connections to other workers in the computation.
    static int32_t RPC_TIMEOUT;
    int32_t id_counter = 0;

    template<typename Q, typename R>
    rpc::Future* _send(int32_t worker_id,
                       void (CWorker::*pLocalFunc)(const Q&, R*),
                       rpc::Future* (spartan::WorkerProxy::*pFunc)(const Q&, const rpc::FutureAttr&),
                       Q& req, R* resp, bool wait=true, int32_t timeout=CBlobCtx::RPC_TIMEOUT) const {

        rpc::Future* fu = NULL;
        if (worker_id == this->worker_id) {
            (local_worker->*pLocalFunc)(req, resp);
        } else {
            std::unordered_map<int32_t, spartan::WorkerProxy*>::iterator it = workers->find(worker_id);

            if (it == workers->end()) {
                Log_error("Cannot find worker_id:%d", worker_id);
                return NULL;
            }

            rpc::FutureAttr fu_attr;
            fu_attr.callback = [resp] (rpc::Future* fu) {
                if (fu->get_error_code() == 0) {
                    if (resp != NULL)
                        fu->get_reply() >> *resp;
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
    rpc::Future* _py_send(int32_t worker_id,
                          void (CWorker::*pLocalFunc)(const Q&, R*),
                          rpc::Future* (spartan::WorkerProxy::*pFunc)(const Q&, const rpc::FutureAttr&),
                          Q& req, R* resp) const {

        rpc::Future* fu = NULL;
        if (worker_id == this->worker_id) {
            (local_worker->*pLocalFunc)(req, resp);
        } else {
            std::unordered_map<int32_t, spartan::WorkerProxy*>::iterator it = workers->find(worker_id);

            if (it == workers->end()) {
                Log_error("Cannot find worker_id:%d", worker_id);
            } else {
                rpc::FutureAttr fu_attr;
                fu = ((it->second)->*pFunc)(req, fu_attr);
                if (fu == NULL) {
                    Log_error("cannot connect to worker:%d", worker_id);
                }
            }
        }
        return fu;
    }

    template<typename Q, typename R>
    void _send_all(rpc::Future* (spartan::WorkerProxy::*pFunc)(const Q&, const rpc::FutureAttr&),
                               Q& req, std::vector<R*>* resp,
                               std::vector<int32_t>* target_workers=NULL,
                               bool wait=true, int32_t timeout=CBlobCtx::RPC_TIMEOUT) const {
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
    CBlobCtx(int32_t wid, std::unordered_map<int32_t, spartan::WorkerProxy*>* peers, CWorker* l_worker = NULL):
        worker_id(wid), local_worker(l_worker), num_workers(peers->size()), workers(peers) {
    }

    ~CBlobCtx() {}

    inline bool is_master() const {
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

    rpc::Future* get(const TileId& tile_id, const CSliceIdx& subslice, GetResp* resp,
                     bool wait=true, int32_t timeout=CBlobCtx::RPC_TIMEOUT) {
        GetReq req(tile_id, subslice);
        return _send<GetReq, GetResp>(tile_id.worker, &CWorker::get,
                                      &spartan::WorkerProxy::async_get, req, resp, wait, timeout);
    }

    unsigned long py_get(TileId* tile_id, CSliceIdx* subslice, GetResp* resp) {
        GetReq req(*tile_id, *subslice);
        return (unsigned long)_py_send<GetReq, GetResp>(tile_id->worker, &CWorker::get,
                                         &spartan::WorkerProxy::async_get, req, resp);
    }

    rpc::Future* get_flatten(const TileId& tile_id, const CSliceIdx& subslice, GetResp* resp,
            bool wait=true, int32_t timeout=CBlobCtx::RPC_TIMEOUT) {
        GetReq req(tile_id, subslice);
        return _send<GetReq, GetResp>(tile_id.worker, &CWorker::get_flatten,
                                      &spartan::WorkerProxy::async_get_flatten,
                                      req, resp, wait, timeout);
    }

    unsigned long py_get_flatten(TileId* tile_id, CSliceIdx* subslice, GetResp* resp) {
        GetReq req(*tile_id, *subslice);
        return (unsigned long)_py_send<GetReq, GetResp>(tile_id->worker, &CWorker::get_flatten,
                                         &spartan::WorkerProxy::async_get_flatten, req, resp);
    }

    rpc::Future* create(const CTile& data, int32_t hint, TileIdMessage* resp, int32_t timeout=CBlobCtx::RPC_TIMEOUT) {
        int32_t w_id;
        if (is_master()) {
            w_id = (hint < 0)? (id_counter++) % num_workers : hint % num_workers;

            while (workers->find(w_id) == workers->end()) {
                w_id = rand() % num_workers;
            }
        } else {
          w_id = worker_id;
        }

        TileId tile_id(w_id, -1);
        CreateTileReq req(tile_id, data);
        return _send<CreateTileReq, TileIdMessage>(w_id, &CWorker::create,
                                                   &spartan::WorkerProxy::async_create,
                                                   req, resp, false, timeout);
    }

    unsigned long py_create(CTile* data, TileIdMessage* resp) {
        TileId tile_id(worker_id, -1);
        CreateTileReq req(tile_id, *data);
        local_worker->create(req, resp);
        return 0;
    }

    rpc::Future* update(const TileId& tile_id, const CSliceIdx& region, std::string& data,
                        int reducer, bool wait=true, int32_t timeout=CBlobCtx::RPC_TIMEOUT) {
        UpdateReq req(tile_id, region, data, reducer);
        return _send<UpdateReq, EmptyMessage>(tile_id.worker, &CWorker::update,
                                              &spartan::WorkerProxy::async_update,
                                              req, NULL, wait, timeout);
    }

    unsigned long py_update(TileId* tile_id, CSliceIdx* region, std::string* data,
                        int reducer) {
        UpdateReq req(*tile_id, *region, *data, reducer);
        return (unsigned long)_py_send<UpdateReq, EmptyMessage>(tile_id->worker, &CWorker::update,
                                                 &spartan::WorkerProxy::async_update,
                                                 req, NULL);
    }

    int8_t cancel_tile(const TileId& tile_id) {
        TileIdMessage req(tile_id);
        int8_t resp;
        _send<TileIdMessage, int8_t>(tile_id.worker, &CWorker::cancel_tile,
                              &spartan::WorkerProxy::async_cancel_tile, req, &resp);
        return resp;
    }

    TileInfoResp* get_tile_info(const TileId& tile_id) {
        TileIdMessage req(tile_id);
        TileInfoResp* resp = new TileInfoResp();
        _send<TileIdMessage, TileInfoResp>(tile_id.worker, &CWorker::get_tile_info,
                             &spartan::WorkerProxy::async_get_tile_info, req, resp);
        return resp;
    }

    std::vector<RunKernelResp*>* map(std::vector<TileId>& tile_ids, std::string& mapper_fn) {
        std::vector<RunKernelResp*>* resp = new std::vector<RunKernelResp*>();
        RunKernelReq req(tile_ids, mapper_fn);
        _send_all<RunKernelReq, RunKernelResp>(&spartan::WorkerProxy::async_run_kernel,
                                               req, resp);
        return resp;
    }

};

int32_t CBlobCtx::RPC_TIMEOUT = 1000;

#endif
