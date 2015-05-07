#include "cblob_ctx.h"

template<typename Q, typename R>
rpc::Future* CBlobCtx::_send(int32_t worker_id, void (CWorker::*pLocalFunc)(const Q&, R*),
                            rpc::Future* (spartan::WorkerProxy::*pFunc)(const Q&, const rpc::FutureAttr&),
                            Q& req, R* resp, bool wait, int32_t timeout) const
{

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
rpc::Future* CBlobCtx::_py_send(int32_t worker_id, void (CWorker::*pLocalFunc)(const Q&, R*),
                                rpc::Future* (spartan::WorkerProxy::*pFunc)(const Q&, const rpc::FutureAttr&),
                                Q& req, R* resp) const
{
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
void CBlobCtx::_send_all(rpc::Future* (spartan::WorkerProxy::*pFunc)(const Q&, const rpc::FutureAttr&),
                         Q& req, std::vector<R*>* resp, std::vector<int32_t>* target_workers,
                         bool wait, int32_t timeout) const
{
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
