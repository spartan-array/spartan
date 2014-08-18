#pragma once

#include "rpc/server.h"
#include "rpc/client.h"

#include <errno.h>


namespace rlog {

class RLogService: public rpc::Service {
public:
    enum {
        LOG = 0x57982f30,
        AGGREGATE_QPS = 0x5d6a8636,
    };
    int __reg_to__(rpc::Server* svr) {
        int ret = 0;
        if ((ret = svr->reg(LOG, this, &RLogService::__log__wrapper__)) != 0) {
            goto err;
        }
        if ((ret = svr->reg(AGGREGATE_QPS, this, &RLogService::__aggregate_qps__wrapper__)) != 0) {
            goto err;
        }
        return 0;
    err:
        svr->unreg(LOG);
        svr->unreg(AGGREGATE_QPS);
        return ret;
    }
    // these RPC handler functions need to be implemented by user
    // for 'raw' handlers, remember to reply req, delete req, and sconn->release(); use sconn->run_async for heavy job
    virtual void log(const rpc::i32& level, const std::string& source, const rpc::i64& msg_id, const std::string& message) = 0;
    virtual void aggregate_qps(const std::string& metric_name, const rpc::i32& increment) = 0;
private:
    void __log__wrapper__(rpc::Request* req, rpc::ServerConnection* sconn) {
        auto f = [=] {
            rpc::i32 in_0;
            req->m >> in_0;
            std::string in_1;
            req->m >> in_1;
            rpc::i64 in_2;
            req->m >> in_2;
            std::string in_3;
            req->m >> in_3;
            this->log(in_0, in_1, in_2, in_3);
            sconn->begin_reply(req);
            sconn->end_reply();
            delete req;
            sconn->release();
        };
        sconn->run_async(f);
    }
    void __aggregate_qps__wrapper__(rpc::Request* req, rpc::ServerConnection* sconn) {
        auto f = [=] {
            std::string in_0;
            req->m >> in_0;
            rpc::i32 in_1;
            req->m >> in_1;
            this->aggregate_qps(in_0, in_1);
            sconn->begin_reply(req);
            sconn->end_reply();
            delete req;
            sconn->release();
        };
        sconn->run_async(f);
    }
};

class RLogProxy {
protected:
    rpc::Client* __cl__;
public:
    RLogProxy(rpc::Client* cl): __cl__(cl) { }
    rpc::Future* async_log(const rpc::i32& level, const std::string& source, const rpc::i64& msg_id, const std::string& message, const rpc::FutureAttr& __fu_attr__ = rpc::FutureAttr()) {
        rpc::Future* __fu__ = __cl__->begin_request(RLogService::LOG, __fu_attr__);
        if (__fu__ != nullptr) {
            *__cl__ << level;
            *__cl__ << source;
            *__cl__ << msg_id;
            *__cl__ << message;
        }
        __cl__->end_request();
        return __fu__;
    }
    rpc::i32 log(const rpc::i32& level, const std::string& source, const rpc::i64& msg_id, const std::string& message) {
        rpc::Future* __fu__ = this->async_log(level, source, msg_id, message);
        if (__fu__ == nullptr) {
            return ENOTCONN;
        }
        rpc::i32 __ret__ = __fu__->get_error_code();
        __fu__->release();
        return __ret__;
    }
    rpc::Future* async_aggregate_qps(const std::string& metric_name, const rpc::i32& increment, const rpc::FutureAttr& __fu_attr__ = rpc::FutureAttr()) {
        rpc::Future* __fu__ = __cl__->begin_request(RLogService::AGGREGATE_QPS, __fu_attr__);
        if (__fu__ != nullptr) {
            *__cl__ << metric_name;
            *__cl__ << increment;
        }
        __cl__->end_request();
        return __fu__;
    }
    rpc::i32 aggregate_qps(const std::string& metric_name, const rpc::i32& increment) {
        rpc::Future* __fu__ = this->async_aggregate_qps(metric_name, increment);
        if (__fu__ == nullptr) {
            return ENOTCONN;
        }
        rpc::i32 __ret__ = __fu__->get_error_code();
        __fu__->release();
        return __ret__;
    }
};

} // namespace rlog



