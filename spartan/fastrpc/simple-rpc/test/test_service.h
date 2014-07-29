#pragma once

#include "rpc/server.h"
#include "rpc/client.h"

#include <errno.h>


namespace test {

struct empty_struct {
};

inline rpc::Marshal& operator <<(rpc::Marshal& m, const empty_struct& o) {
    return m;
}

inline rpc::Marshal& operator >>(rpc::Marshal& m, empty_struct& o) {
    return m;
}

struct complex_struct {
    std::map<std::pair<std::string, std::string>, std::vector<std::vector<std::pair<std::string, std::string>>>> d;
    std::set<std::string> s;
    empty_struct e;
};

inline rpc::Marshal& operator <<(rpc::Marshal& m, const complex_struct& o) {
    m << o.d;
    m << o.s;
    m << o.e;
    return m;
}

inline rpc::Marshal& operator >>(rpc::Marshal& m, complex_struct& o) {
    m >> o.d;
    m >> o.s;
    m >> o.e;
    return m;
}

class EmptyService: public rpc::Service {
public:
    enum {
    };
    int __reg_to__(rpc::Server* svr) {
        int ret = 0;
        return 0;
    err:
        return ret;
    }
    // these RPC handler functions need to be implemented by user
    // for 'raw' handlers, remember to reply req, delete req, and sconn->release(); use sconn->run_async for heavy job
private:
};

class EmptyProxy {
protected:
    rpc::Client* __cl__;
public:
    EmptyProxy(rpc::Client* cl): __cl__(cl) { }
};

class MathService: public rpc::Service {
public:
    enum {
        GCD = 0x475dd711,
    };
    int __reg_to__(rpc::Server* svr) {
        int ret = 0;
        if ((ret = svr->reg(GCD, this, &MathService::__gcd__wrapper__)) != 0) {
            goto err;
        }
        return 0;
    err:
        svr->unreg(GCD);
        return ret;
    }
    // these RPC handler functions need to be implemented by user
    // for 'raw' handlers, remember to reply req, delete req, and sconn->release(); use sconn->run_async for heavy job
    virtual void gcd(const rpc::i64& a, const rpc::i64&, rpc::i64* g);
private:
    void __gcd__wrapper__(rpc::Request* req, rpc::ServerConnection* sconn) {
        auto f = [=] {
            rpc::i64 in_0;
            req->m >> in_0;
            rpc::i64 in_1;
            req->m >> in_1;
            rpc::i64 out_0;
            this->gcd(in_0, in_1, &out_0);
            sconn->begin_reply(req);
            *sconn << out_0;
            sconn->end_reply();
            delete req;
            sconn->release();
        };
        sconn->run_async(f);
    }
};

class MathProxy {
protected:
    rpc::Client* __cl__;
public:
    MathProxy(rpc::Client* cl): __cl__(cl) { }
    rpc::Future* async_gcd(const rpc::i64& a, const rpc::i64& in_1, const rpc::FutureAttr& __fu_attr__ = rpc::FutureAttr()) {
        rpc::Future* __fu__ = __cl__->begin_request(MathService::GCD, __fu_attr__);
        if (__fu__ != nullptr) {
            *__cl__ << a;
            *__cl__ << in_1;
        }
        __cl__->end_request();
        return __fu__;
    }
    rpc::i32 gcd(const rpc::i64& a, const rpc::i64& in_1, rpc::i64* g) {
        rpc::Future* __fu__ = this->async_gcd(a, in_1);
        if (__fu__ == nullptr) {
            return ENOTCONN;
        }
        rpc::i32 __ret__ = __fu__->get_error_code();
        if (__ret__ == 0) {
            __fu__->get_reply() >> *g;
        }
        __fu__->release();
        return __ret__;
    }
};

} // namespace test



