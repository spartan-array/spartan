// generated from 'spartan_service.rpc'

#pragma once

#include "rpc/server.h"
#include "rpc/client.h"

#include <errno.h>

#include "spartan/refptr.h"

extern rpc::Marshal& operator <<(rpc::Marshal& m, const RefPtr& p);
extern rpc::Marshal& operator >>(rpc::Marshal& m, RefPtr& p);
namespace spartan {

struct TypeConstructor {
    rpc::i32 type_id;
    std::string opts;
};

inline rpc::Marshal& operator <<(rpc::Marshal& m, const TypeConstructor& o) {
    m << o.type_id;
    m << o.opts;
    return m;
}

inline rpc::Marshal& operator >>(rpc::Marshal& m, TypeConstructor& o) {
    m >> o.type_id;
    m >> o.opts;
    return m;
}

struct CreateTableReq {
    rpc::i32 num_shards;
    std::vector<rpc::i32> shards;
    rpc::i32 id;
    TypeConstructor sharder;
    TypeConstructor combiner;
    TypeConstructor reducer;
    TypeConstructor selector;
};

inline rpc::Marshal& operator <<(rpc::Marshal& m, const CreateTableReq& o) {
    m << o.num_shards;
    m << o.shards;
    m << o.id;
    m << o.sharder;
    m << o.combiner;
    m << o.reducer;
    m << o.selector;
    return m;
}

inline rpc::Marshal& operator >>(rpc::Marshal& m, CreateTableReq& o) {
    m >> o.num_shards;
    m >> o.shards;
    m >> o.id;
    m >> o.sharder;
    m >> o.combiner;
    m >> o.reducer;
    m >> o.selector;
    return m;
}

struct RunKernelReq {
    rpc::i32 kernel;
    rpc::i32 table;
    rpc::i32 shard;
    std::map<std::string, std::string> kernel_args;
    std::map<std::string, std::string> task_args;
};

inline rpc::Marshal& operator <<(rpc::Marshal& m, const RunKernelReq& o) {
    m << o.kernel;
    m << o.table;
    m << o.shard;
    m << o.kernel_args;
    m << o.task_args;
    return m;
}

inline rpc::Marshal& operator >>(rpc::Marshal& m, RunKernelReq& o) {
    m >> o.kernel;
    m >> o.table;
    m >> o.shard;
    m >> o.kernel_args;
    m >> o.task_args;
    return m;
}

struct RunKernelResp {
    double elapsed;
    std::string error;
};

inline rpc::Marshal& operator <<(rpc::Marshal& m, const RunKernelResp& o) {
    m << o.elapsed;
    m << o.error;
    return m;
}

inline rpc::Marshal& operator >>(rpc::Marshal& m, RunKernelResp& o) {
    m >> o.elapsed;
    m >> o.error;
    return m;
}

struct KV {
    RefPtr key;
    RefPtr value;
};

inline rpc::Marshal& operator <<(rpc::Marshal& m, const KV& o) {
    m << o.key;
    m << o.value;
    return m;
}

inline rpc::Marshal& operator >>(rpc::Marshal& m, KV& o) {
    m >> o.key;
    m >> o.value;
    return m;
}

struct IteratorReq {
    rpc::i32 table;
    rpc::i32 shard;
    rpc::i32 id;
    rpc::i32 count;
};

inline rpc::Marshal& operator <<(rpc::Marshal& m, const IteratorReq& o) {
    m << o.table;
    m << o.shard;
    m << o.id;
    m << o.count;
    return m;
}

inline rpc::Marshal& operator >>(rpc::Marshal& m, IteratorReq& o) {
    m >> o.table;
    m >> o.shard;
    m >> o.id;
    m >> o.count;
    return m;
}

struct IteratorResp {
    rpc::i32 id;
    rpc::i32 done;
    std::vector<KV> results;
    rpc::i32 row_count;
};

inline rpc::Marshal& operator <<(rpc::Marshal& m, const IteratorResp& o) {
    m << o.id;
    m << o.done;
    m << o.results;
    m << o.row_count;
    return m;
}

inline rpc::Marshal& operator >>(rpc::Marshal& m, IteratorResp& o) {
    m >> o.id;
    m >> o.done;
    m >> o.results;
    m >> o.row_count;
    return m;
}

struct PartitionInfo {
    rpc::i32 table;
    rpc::i32 shard;
    rpc::i64 entries;
    rpc::i32 owner;
    rpc::i32 dirty;
    rpc::i32 tainted;
};

inline rpc::Marshal& operator <<(rpc::Marshal& m, const PartitionInfo& o) {
    m << o.table;
    m << o.shard;
    m << o.entries;
    m << o.owner;
    m << o.dirty;
    m << o.tainted;
    return m;
}

inline rpc::Marshal& operator >>(rpc::Marshal& m, PartitionInfo& o) {
    m >> o.table;
    m >> o.shard;
    m >> o.entries;
    m >> o.owner;
    m >> o.dirty;
    m >> o.tainted;
    return m;
}

struct TableData {
    rpc::i32 source;
    rpc::i32 table;
    rpc::i32 shard;
    rpc::i32 done;
    std::vector<KV> kv_data;
    rpc::i32 marker;
    rpc::i32 missing_key;
    std::string error;
};

inline rpc::Marshal& operator <<(rpc::Marshal& m, const TableData& o) {
    m << o.source;
    m << o.table;
    m << o.shard;
    m << o.done;
    m << o.kv_data;
    m << o.marker;
    m << o.missing_key;
    m << o.error;
    return m;
}

inline rpc::Marshal& operator >>(rpc::Marshal& m, TableData& o) {
    m >> o.source;
    m >> o.table;
    m >> o.shard;
    m >> o.done;
    m >> o.kv_data;
    m >> o.marker;
    m >> o.missing_key;
    m >> o.error;
    return m;
}

struct GetRequest {
    RefPtr key;
    rpc::i32 table;
    rpc::i32 shard;
};

inline rpc::Marshal& operator <<(rpc::Marshal& m, const GetRequest& o) {
    m << o.key;
    m << o.table;
    m << o.shard;
    return m;
}

inline rpc::Marshal& operator >>(rpc::Marshal& m, GetRequest& o) {
    m >> o.key;
    m >> o.table;
    m >> o.shard;
    return m;
}

struct HostPort {
    std::string host;
    rpc::i32 port;
};

inline rpc::Marshal& operator <<(rpc::Marshal& m, const HostPort& o) {
    m << o.host;
    m << o.port;
    return m;
}

inline rpc::Marshal& operator >>(rpc::Marshal& m, HostPort& o) {
    m >> o.host;
    m >> o.port;
    return m;
}

struct ConfigData {
    HostPort master;
    std::vector<HostPort> workers;
};

inline rpc::Marshal& operator <<(rpc::Marshal& m, const ConfigData& o) {
    m << o.master;
    m << o.workers;
    return m;
}

inline rpc::Marshal& operator >>(rpc::Marshal& m, ConfigData& o) {
    m >> o.master;
    m >> o.workers;
    return m;
}

struct ShardAssignment {
    rpc::i32 table;
    rpc::i32 shard;
    rpc::i32 worker;
};

inline rpc::Marshal& operator <<(rpc::Marshal& m, const ShardAssignment& o) {
    m << o.table;
    m << o.shard;
    m << o.worker;
    return m;
}

inline rpc::Marshal& operator >>(rpc::Marshal& m, ShardAssignment& o) {
    m >> o.table;
    m >> o.shard;
    m >> o.worker;
    return m;
}

struct ShardAssignmentReq {
    std::vector<ShardAssignment> assign;
};

inline rpc::Marshal& operator <<(rpc::Marshal& m, const ShardAssignmentReq& o) {
    m << o.assign;
    return m;
}

inline rpc::Marshal& operator >>(rpc::Marshal& m, ShardAssignmentReq& o) {
    m >> o.assign;
    return m;
}

struct RegisterReq {
    HostPort addr;
};

inline rpc::Marshal& operator <<(rpc::Marshal& m, const RegisterReq& o) {
    m << o.addr;
    return m;
}

inline rpc::Marshal& operator >>(rpc::Marshal& m, RegisterReq& o) {
    m >> o.addr;
    return m;
}

struct WorkerInitReq {
    rpc::i32 id;
    std::map<rpc::i32, HostPort> workers;
};

inline rpc::Marshal& operator <<(rpc::Marshal& m, const WorkerInitReq& o) {
    m << o.id;
    m << o.workers;
    return m;
}

inline rpc::Marshal& operator >>(rpc::Marshal& m, WorkerInitReq& o) {
    m >> o.id;
    m >> o.workers;
    return m;
}

class WorkerService: public rpc::Service {
public:
    enum {
        INITIALIZE = 0x0,
        CREATE_TABLE = 0x1,
        DESTROY_TABLE = 0x2,
        GET = 0x3,
        ASSIGN_SHARDS = 0x4,
        RUN_KERNEL = 0x5,
        GET_ITERATOR = 0x6,
        PUT = 0x7,
        FLUSH = 0x8,
        SHUTDOWN = 0x9,
    };
    int reg_to(rpc::Server* svr) {
        int ret = 0;
        if ((ret = svr->reg(INITIALIZE, this, &WorkerService::__initialize__wrapper__)) != 0) {
            goto err;
        }
        if ((ret = svr->reg(CREATE_TABLE, this, &WorkerService::__create_table__wrapper__)) != 0) {
            goto err;
        }
        if ((ret = svr->reg(DESTROY_TABLE, this, &WorkerService::__destroy_table__wrapper__)) != 0) {
            goto err;
        }
        if ((ret = svr->reg(GET, this, &WorkerService::__get__wrapper__)) != 0) {
            goto err;
        }
        if ((ret = svr->reg(ASSIGN_SHARDS, this, &WorkerService::__assign_shards__wrapper__)) != 0) {
            goto err;
        }
        if ((ret = svr->reg(RUN_KERNEL, this, &WorkerService::__run_kernel__wrapper__)) != 0) {
            goto err;
        }
        if ((ret = svr->reg(GET_ITERATOR, this, &WorkerService::__get_iterator__wrapper__)) != 0) {
            goto err;
        }
        if ((ret = svr->reg(PUT, this, &WorkerService::__put__wrapper__)) != 0) {
            goto err;
        }
        if ((ret = svr->reg(FLUSH, this, &WorkerService::__flush__wrapper__)) != 0) {
            goto err;
        }
        if ((ret = svr->reg(SHUTDOWN, this, &WorkerService::__shutdown__wrapper__)) != 0) {
            goto err;
        }
        return 0;
    err:
        svr->unreg(INITIALIZE);
        svr->unreg(CREATE_TABLE);
        svr->unreg(DESTROY_TABLE);
        svr->unreg(GET);
        svr->unreg(ASSIGN_SHARDS);
        svr->unreg(RUN_KERNEL);
        svr->unreg(GET_ITERATOR);
        svr->unreg(PUT);
        svr->unreg(FLUSH);
        svr->unreg(SHUTDOWN);
        return ret;
    }
    // these RPC handler functions need to be implemented by user
    // for 'raw' handlers, remember to reply req, delete req, and sconn->release(); use sconn->run_async for heavy job
    virtual void initialize(const WorkerInitReq& req) = 0;
    virtual void create_table(const CreateTableReq& req) = 0;
    virtual void destroy_table(const rpc::i32& table_id) = 0;
    virtual void get(const GetRequest& req, TableData* resp) = 0;
    virtual void assign_shards(const ShardAssignmentReq& req) = 0;
    virtual void run_kernel(const RunKernelReq& req, RunKernelResp* resp) = 0;
    virtual void get_iterator(const IteratorReq& req, IteratorResp* resp) = 0;
    virtual void put(const TableData& req) = 0;
    virtual void flush() = 0;
    virtual void shutdown() = 0;
private:
    void __initialize__wrapper__(rpc::Request* req, rpc::ServerConnection* sconn) {
        auto f = [=] {
            WorkerInitReq in_0;
            req->m >> in_0;
            this->initialize(in_0);
            sconn->begin_reply(req);
            sconn->end_reply();
            delete req;
            sconn->release();
        };
        sconn->run_async(f);
    }
    void __create_table__wrapper__(rpc::Request* req, rpc::ServerConnection* sconn) {
        auto f = [=] {
            CreateTableReq in_0;
            req->m >> in_0;
            this->create_table(in_0);
            sconn->begin_reply(req);
            sconn->end_reply();
            delete req;
            sconn->release();
        };
        sconn->run_async(f);
    }
    void __destroy_table__wrapper__(rpc::Request* req, rpc::ServerConnection* sconn) {
        auto f = [=] {
            rpc::i32 in_0;
            req->m >> in_0;
            this->destroy_table(in_0);
            sconn->begin_reply(req);
            sconn->end_reply();
            delete req;
            sconn->release();
        };
        sconn->run_async(f);
    }
    void __get__wrapper__(rpc::Request* req, rpc::ServerConnection* sconn) {
        auto f = [=] {
            GetRequest in_0;
            req->m >> in_0;
            TableData out_0;
            this->get(in_0, &out_0);
            sconn->begin_reply(req);
            *sconn << out_0;
            sconn->end_reply();
            delete req;
            sconn->release();
        };
        sconn->run_async(f);
    }
    void __assign_shards__wrapper__(rpc::Request* req, rpc::ServerConnection* sconn) {
        auto f = [=] {
            ShardAssignmentReq in_0;
            req->m >> in_0;
            this->assign_shards(in_0);
            sconn->begin_reply(req);
            sconn->end_reply();
            delete req;
            sconn->release();
        };
        sconn->run_async(f);
    }
    void __run_kernel__wrapper__(rpc::Request* req, rpc::ServerConnection* sconn) {
        auto f = [=] {
            RunKernelReq in_0;
            req->m >> in_0;
            RunKernelResp out_0;
            this->run_kernel(in_0, &out_0);
            sconn->begin_reply(req);
            *sconn << out_0;
            sconn->end_reply();
            delete req;
            sconn->release();
        };
        sconn->run_async(f);
    }
    void __get_iterator__wrapper__(rpc::Request* req, rpc::ServerConnection* sconn) {
        auto f = [=] {
            IteratorReq in_0;
            req->m >> in_0;
            IteratorResp out_0;
            this->get_iterator(in_0, &out_0);
            sconn->begin_reply(req);
            *sconn << out_0;
            sconn->end_reply();
            delete req;
            sconn->release();
        };
        sconn->run_async(f);
    }
    void __put__wrapper__(rpc::Request* req, rpc::ServerConnection* sconn) {
        auto f = [=] {
            TableData in_0;
            req->m >> in_0;
            this->put(in_0);
            sconn->begin_reply(req);
            sconn->end_reply();
            delete req;
            sconn->release();
        };
        sconn->run_async(f);
    }
    void __flush__wrapper__(rpc::Request* req, rpc::ServerConnection* sconn) {
        auto f = [=] {
            this->flush();
            sconn->begin_reply(req);
            sconn->end_reply();
            delete req;
            sconn->release();
        };
        sconn->run_async(f);
    }
    void __shutdown__wrapper__(rpc::Request* req, rpc::ServerConnection* sconn) {
        auto f = [=] {
            this->shutdown();
            sconn->begin_reply(req);
            sconn->end_reply();
            delete req;
            sconn->release();
        };
        sconn->run_async(f);
    }
};

class WorkerProxy {
protected:
    rpc::Client* __cl__;
public:
    WorkerProxy(rpc::Client* cl): __cl__(cl) { }
    rpc::Future* async_initialize(const WorkerInitReq& req, const rpc::FutureAttr& __fu_attr__ = rpc::FutureAttr()) {
        rpc::Future* __fu__ = __cl__->begin_request(WorkerService::INITIALIZE, __fu_attr__);
        if (__fu__ != nullptr) {
            *__cl__ << req;
        }
        __cl__->end_request();
        return __fu__;
    }
    rpc::i32 initialize(const WorkerInitReq& req) {
        rpc::Future* __fu__ = this->async_initialize(req);
        if (__fu__ == nullptr) {
            return ENOTCONN;
        }
        rpc::i32 __ret__ = __fu__->get_error_code();
        __fu__->release();
        return __ret__;
    }
    rpc::Future* async_create_table(const CreateTableReq& req, const rpc::FutureAttr& __fu_attr__ = rpc::FutureAttr()) {
        rpc::Future* __fu__ = __cl__->begin_request(WorkerService::CREATE_TABLE, __fu_attr__);
        if (__fu__ != nullptr) {
            *__cl__ << req;
        }
        __cl__->end_request();
        return __fu__;
    }
    rpc::i32 create_table(const CreateTableReq& req) {
        rpc::Future* __fu__ = this->async_create_table(req);
        if (__fu__ == nullptr) {
            return ENOTCONN;
        }
        rpc::i32 __ret__ = __fu__->get_error_code();
        __fu__->release();
        return __ret__;
    }
    rpc::Future* async_destroy_table(const rpc::i32& table_id, const rpc::FutureAttr& __fu_attr__ = rpc::FutureAttr()) {
        rpc::Future* __fu__ = __cl__->begin_request(WorkerService::DESTROY_TABLE, __fu_attr__);
        if (__fu__ != nullptr) {
            *__cl__ << table_id;
        }
        __cl__->end_request();
        return __fu__;
    }
    rpc::i32 destroy_table(const rpc::i32& table_id) {
        rpc::Future* __fu__ = this->async_destroy_table(table_id);
        if (__fu__ == nullptr) {
            return ENOTCONN;
        }
        rpc::i32 __ret__ = __fu__->get_error_code();
        __fu__->release();
        return __ret__;
    }
    rpc::Future* async_get(const GetRequest& req, const rpc::FutureAttr& __fu_attr__ = rpc::FutureAttr()) {
        rpc::Future* __fu__ = __cl__->begin_request(WorkerService::GET, __fu_attr__);
        if (__fu__ != nullptr) {
            *__cl__ << req;
        }
        __cl__->end_request();
        return __fu__;
    }
    rpc::i32 get(const GetRequest& req, TableData* resp) {
        rpc::Future* __fu__ = this->async_get(req);
        if (__fu__ == nullptr) {
            return ENOTCONN;
        }
        rpc::i32 __ret__ = __fu__->get_error_code();
        if (__ret__ == 0) {
            __fu__->get_reply() >> *resp;
        }
        __fu__->release();
        return __ret__;
    }
    rpc::Future* async_assign_shards(const ShardAssignmentReq& req, const rpc::FutureAttr& __fu_attr__ = rpc::FutureAttr()) {
        rpc::Future* __fu__ = __cl__->begin_request(WorkerService::ASSIGN_SHARDS, __fu_attr__);
        if (__fu__ != nullptr) {
            *__cl__ << req;
        }
        __cl__->end_request();
        return __fu__;
    }
    rpc::i32 assign_shards(const ShardAssignmentReq& req) {
        rpc::Future* __fu__ = this->async_assign_shards(req);
        if (__fu__ == nullptr) {
            return ENOTCONN;
        }
        rpc::i32 __ret__ = __fu__->get_error_code();
        __fu__->release();
        return __ret__;
    }
    rpc::Future* async_run_kernel(const RunKernelReq& req, const rpc::FutureAttr& __fu_attr__ = rpc::FutureAttr()) {
        rpc::Future* __fu__ = __cl__->begin_request(WorkerService::RUN_KERNEL, __fu_attr__);
        if (__fu__ != nullptr) {
            *__cl__ << req;
        }
        __cl__->end_request();
        return __fu__;
    }
    rpc::i32 run_kernel(const RunKernelReq& req, RunKernelResp* resp) {
        rpc::Future* __fu__ = this->async_run_kernel(req);
        if (__fu__ == nullptr) {
            return ENOTCONN;
        }
        rpc::i32 __ret__ = __fu__->get_error_code();
        if (__ret__ == 0) {
            __fu__->get_reply() >> *resp;
        }
        __fu__->release();
        return __ret__;
    }
    rpc::Future* async_get_iterator(const IteratorReq& req, const rpc::FutureAttr& __fu_attr__ = rpc::FutureAttr()) {
        rpc::Future* __fu__ = __cl__->begin_request(WorkerService::GET_ITERATOR, __fu_attr__);
        if (__fu__ != nullptr) {
            *__cl__ << req;
        }
        __cl__->end_request();
        return __fu__;
    }
    rpc::i32 get_iterator(const IteratorReq& req, IteratorResp* resp) {
        rpc::Future* __fu__ = this->async_get_iterator(req);
        if (__fu__ == nullptr) {
            return ENOTCONN;
        }
        rpc::i32 __ret__ = __fu__->get_error_code();
        if (__ret__ == 0) {
            __fu__->get_reply() >> *resp;
        }
        __fu__->release();
        return __ret__;
    }
    rpc::Future* async_put(const TableData& req, const rpc::FutureAttr& __fu_attr__ = rpc::FutureAttr()) {
        rpc::Future* __fu__ = __cl__->begin_request(WorkerService::PUT, __fu_attr__);
        if (__fu__ != nullptr) {
            *__cl__ << req;
        }
        __cl__->end_request();
        return __fu__;
    }
    rpc::i32 put(const TableData& req) {
        rpc::Future* __fu__ = this->async_put(req);
        if (__fu__ == nullptr) {
            return ENOTCONN;
        }
        rpc::i32 __ret__ = __fu__->get_error_code();
        __fu__->release();
        return __ret__;
    }
    rpc::Future* async_flush(const rpc::FutureAttr& __fu_attr__ = rpc::FutureAttr()) {
        rpc::Future* __fu__ = __cl__->begin_request(WorkerService::FLUSH, __fu_attr__);
        __cl__->end_request();
        return __fu__;
    }
    rpc::i32 flush() {
        rpc::Future* __fu__ = this->async_flush();
        if (__fu__ == nullptr) {
            return ENOTCONN;
        }
        rpc::i32 __ret__ = __fu__->get_error_code();
        __fu__->release();
        return __ret__;
    }
    rpc::Future* async_shutdown(const rpc::FutureAttr& __fu_attr__ = rpc::FutureAttr()) {
        rpc::Future* __fu__ = __cl__->begin_request(WorkerService::SHUTDOWN, __fu_attr__);
        __cl__->end_request();
        return __fu__;
    }
    rpc::i32 shutdown() {
        rpc::Future* __fu__ = this->async_shutdown();
        if (__fu__ == nullptr) {
            return ENOTCONN;
        }
        rpc::i32 __ret__ = __fu__->get_error_code();
        __fu__->release();
        return __ret__;
    }
};

class MasterService: public rpc::Service {
public:
    enum {
        REGISTER_WORKER = 0x0,
    };
    int reg_to(rpc::Server* svr) {
        int ret = 0;
        if ((ret = svr->reg(REGISTER_WORKER, this, &MasterService::__register_worker__wrapper__)) != 0) {
            goto err;
        }
        return 0;
    err:
        svr->unreg(REGISTER_WORKER);
        return ret;
    }
    // these RPC handler functions need to be implemented by user
    // for 'raw' handlers, remember to reply req, delete req, and sconn->release(); use sconn->run_async for heavy job
    virtual void register_worker(const RegisterReq& req) = 0;
private:
    void __register_worker__wrapper__(rpc::Request* req, rpc::ServerConnection* sconn) {
        RegisterReq in_0;
        req->m >> in_0;
        this->register_worker(in_0);
        sconn->begin_reply(req);
        sconn->end_reply();
        delete req;
        sconn->release();
    }
};

class MasterProxy {
protected:
    rpc::Client* __cl__;
public:
    MasterProxy(rpc::Client* cl): __cl__(cl) { }
    rpc::Future* async_register_worker(const RegisterReq& req, const rpc::FutureAttr& __fu_attr__ = rpc::FutureAttr()) {
        rpc::Future* __fu__ = __cl__->begin_request(MasterService::REGISTER_WORKER, __fu_attr__);
        if (__fu__ != nullptr) {
            *__cl__ << req;
        }
        __cl__->end_request();
        return __fu__;
    }
    rpc::i32 register_worker(const RegisterReq& req) {
        rpc::Future* __fu__ = this->async_register_worker(req);
        if (__fu__ == nullptr) {
            return ENOTCONN;
        }
        rpc::i32 __ret__ = __fu__->get_error_code();
        __fu__->release();
        return __ret__;
    }
};

} // namespace spartan

