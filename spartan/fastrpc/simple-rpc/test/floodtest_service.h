#pragma once

#include "rpc/server.h"
#include "rpc/client.h"

#include <errno.h>


namespace floodtest {

class FloodService: public rpc::Service {
public:
    enum {
        UPDATE_NODE_LIST = 0x43102c1b,
        FLOOD = 0x4c64a660,
        FLOOD_UDP = 0x632a2de2,
    };
    int __reg_to__(rpc::Server* svr) {
        int ret = 0;
        if ((ret = svr->reg(UPDATE_NODE_LIST, this, &FloodService::__update_node_list__wrapper__)) != 0) {
            goto err;
        }
        if ((ret = svr->reg(FLOOD, this, &FloodService::__flood__wrapper__)) != 0) {
            goto err;
        }
        svr->enable_udp();
        if ((ret = svr->reg(FLOOD_UDP, this, &FloodService::__flood_udp__wrapper__)) != 0) {
            goto err;
        }
        return 0;
    err:
        svr->unreg(UPDATE_NODE_LIST);
        svr->unreg(FLOOD);
        svr->unreg(FLOOD_UDP);
        return ret;
    }
    // these RPC handler functions need to be implemented by user
    // for 'raw' handlers, remember to reply req, delete req, and sconn->release(); use sconn->run_async for heavy job
    virtual void update_node_list(const std::vector<std::string>& nodes);
    virtual void flood();
    virtual void flood_udp();
private:
    void __update_node_list__wrapper__(rpc::Request* req, rpc::ServerConnection* sconn) {
        std::vector<std::string> in_0;
        req->m >> in_0;
        this->update_node_list(in_0);
        sconn->begin_reply(req);
        sconn->end_reply();
        delete req;
        sconn->release();
    }
    void __flood__wrapper__(rpc::Request* req, rpc::ServerConnection* sconn) {
        this->flood();
        sconn->begin_reply(req);
        sconn->end_reply();
        delete req;
        sconn->release();
    }
    void __flood_udp__wrapper__(rpc::Request* req, rpc::ServerConnection* sconn) {
        this->flood_udp();
        delete req;
        sconn->release();
    }
};

class FloodProxy {
protected:
    rpc::Client* __cl__;
public:
    FloodProxy(rpc::Client* cl): __cl__(cl) { }
    rpc::Future* async_update_node_list(const std::vector<std::string>& nodes, const rpc::FutureAttr& __fu_attr__ = rpc::FutureAttr()) {
        rpc::Future* __fu__ = __cl__->begin_request(FloodService::UPDATE_NODE_LIST, __fu_attr__);
        if (__fu__ != nullptr) {
            *__cl__ << nodes;
        }
        __cl__->end_request();
        return __fu__;
    }
    rpc::i32 update_node_list(const std::vector<std::string>& nodes) {
        rpc::Future* __fu__ = this->async_update_node_list(nodes);
        if (__fu__ == nullptr) {
            return ENOTCONN;
        }
        rpc::i32 __ret__ = __fu__->get_error_code();
        __fu__->release();
        return __ret__;
    }
    rpc::Future* async_flood(const rpc::FutureAttr& __fu_attr__ = rpc::FutureAttr()) {
        rpc::Future* __fu__ = __cl__->begin_request(FloodService::FLOOD, __fu_attr__);
        __cl__->end_request();
        return __fu__;
    }
    rpc::i32 flood() {
        rpc::Future* __fu__ = this->async_flood();
        if (__fu__ == nullptr) {
            return ENOTCONN;
        }
        rpc::i32 __ret__ = __fu__->get_error_code();
        __fu__->release();
        return __ret__;
    }
    int flood_udp() /* UDP */ {
        __cl__->begin_udp_request(FloodService::FLOOD_UDP);
        return __cl__->end_udp_request();
    }
};

} // namespace floodtest



