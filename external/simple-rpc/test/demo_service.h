// generated from 'demo_service.rpc'

#pragma once

#include "rpc/server.h"
#include "rpc/client.h"

#include <errno.h>

namespace demo {

struct point3 {
    double x;
    double y;
    double z;
};

inline rpc::Marshal& operator <<(rpc::Marshal& m, const point3& o) {
    m << o.x;
    m << o.y;
    m << o.z;
    return m;
}

inline rpc::Marshal& operator >>(rpc::Marshal& m, point3& o) {
    m >> o.x;
    m >> o.y;
    m >> o.z;
    return m;
}

class DemoService: public rpc::Service {
public:
    enum {
        FAST_PRIME = 0x667c8434,
        FAST_DOT_PROD = 0x6dfeacf9,
        FAST_LARGE_STR_NOP = 0x1753b1da,
        FAST_VEC_LEN = 0x5d8e0ce6,
        PRIME = 0x27f49cc3,
        DOT_PROD = 0x3a67d81e,
        LARGE_STR_NOP = 0x18482591,
    };
    int reg_to(rpc::Server* svr) {
        int ret = 0;
        if ((ret = svr->reg(FAST_PRIME, this, &DemoService::__fast_prime__wrapper__)) != 0) {
            goto err;
        }
        if ((ret = svr->reg(FAST_DOT_PROD, this, &DemoService::__fast_dot_prod__wrapper__)) != 0) {
            goto err;
        }
        if ((ret = svr->reg(FAST_LARGE_STR_NOP, this, &DemoService::__fast_large_str_nop__wrapper__)) != 0) {
            goto err;
        }
        if ((ret = svr->reg(FAST_VEC_LEN, this, &DemoService::__fast_vec_len__wrapper__)) != 0) {
            goto err;
        }
        if ((ret = svr->reg(PRIME, this, &DemoService::__prime__wrapper__)) != 0) {
            goto err;
        }
        if ((ret = svr->reg(DOT_PROD, this, &DemoService::__dot_prod__wrapper__)) != 0) {
            goto err;
        }
        if ((ret = svr->reg(LARGE_STR_NOP, this, &DemoService::__large_str_nop__wrapper__)) != 0) {
            goto err;
        }
        return 0;
    err:
        svr->unreg(FAST_PRIME);
        svr->unreg(FAST_DOT_PROD);
        svr->unreg(FAST_LARGE_STR_NOP);
        svr->unreg(FAST_VEC_LEN);
        svr->unreg(PRIME);
        svr->unreg(DOT_PROD);
        svr->unreg(LARGE_STR_NOP);
        return ret;
    }
    // these RPC handler functions need to be implemented by user
    // for 'raw' handlers, remember to reply req, delete req, and sconn->release(); use sconn->run_async for heavy job
    virtual void fast_prime(const rpc::i32& n, rpc::i32* flag);
    virtual void fast_dot_prod(const point3& p1, const point3& p2, double* v);
    virtual void fast_large_str_nop(const std::string& str);
    virtual void fast_vec_len(const std::vector<std::vector<std::string>>& v, rpc::i32* len);
    virtual void prime(const rpc::i32& n, rpc::i32* flag);
    virtual void dot_prod(const point3& p1, const point3& p2, double* v);
    virtual void large_str_nop(const std::string& str);
private:
    void __fast_prime__wrapper__(rpc::Request* req, rpc::ServerConnection* sconn) {
        rpc::i32 in_0;
        req->m >> in_0;
        rpc::i32 out_0;
        this->fast_prime(in_0, &out_0);
        sconn->begin_reply(req);
        *sconn << out_0;
        sconn->end_reply();
        delete req;
        sconn->release();
    }
    void __fast_dot_prod__wrapper__(rpc::Request* req, rpc::ServerConnection* sconn) {
        point3 in_0;
        req->m >> in_0;
        point3 in_1;
        req->m >> in_1;
        double out_0;
        this->fast_dot_prod(in_0, in_1, &out_0);
        sconn->begin_reply(req);
        *sconn << out_0;
        sconn->end_reply();
        delete req;
        sconn->release();
    }
    void __fast_large_str_nop__wrapper__(rpc::Request* req, rpc::ServerConnection* sconn) {
        std::string in_0;
        req->m >> in_0;
        this->fast_large_str_nop(in_0);
        sconn->begin_reply(req);
        sconn->end_reply();
        delete req;
        sconn->release();
    }
    void __fast_vec_len__wrapper__(rpc::Request* req, rpc::ServerConnection* sconn) {
        std::vector<std::vector<std::string>> in_0;
        req->m >> in_0;
        rpc::i32 out_0;
        this->fast_vec_len(in_0, &out_0);
        sconn->begin_reply(req);
        *sconn << out_0;
        sconn->end_reply();
        delete req;
        sconn->release();
    }
    void __prime__wrapper__(rpc::Request* req, rpc::ServerConnection* sconn) {
        auto f = [=] {
            rpc::i32 in_0;
            req->m >> in_0;
            rpc::i32 out_0;
            this->prime(in_0, &out_0);
            sconn->begin_reply(req);
            *sconn << out_0;
            sconn->end_reply();
            delete req;
            sconn->release();
        };
        sconn->run_async(f);
    }
    void __dot_prod__wrapper__(rpc::Request* req, rpc::ServerConnection* sconn) {
        auto f = [=] {
            point3 in_0;
            req->m >> in_0;
            point3 in_1;
            req->m >> in_1;
            double out_0;
            this->dot_prod(in_0, in_1, &out_0);
            sconn->begin_reply(req);
            *sconn << out_0;
            sconn->end_reply();
            delete req;
            sconn->release();
        };
        sconn->run_async(f);
    }
    void __large_str_nop__wrapper__(rpc::Request* req, rpc::ServerConnection* sconn) {
        auto f = [=] {
            std::string in_0;
            req->m >> in_0;
            this->large_str_nop(in_0);
            sconn->begin_reply(req);
            sconn->end_reply();
            delete req;
            sconn->release();
        };
        sconn->run_async(f);
    }
};

class DemoProxy {
protected:
    rpc::Client* __cl__;
public:
    DemoProxy(rpc::Client* cl): __cl__(cl) { }
    rpc::Future* async_fast_prime(const rpc::i32& n, const rpc::FutureAttr& __fu_attr__ = rpc::FutureAttr()) {
        rpc::Future* __fu__ = __cl__->begin_request(DemoService::FAST_PRIME, __fu_attr__);
        if (__fu__ != nullptr) {
            *__cl__ << n;
        }
        __cl__->end_request();
        return __fu__;
    }
    rpc::i32 fast_prime(const rpc::i32& n, rpc::i32* flag) {
        rpc::Future* __fu__ = this->async_fast_prime(n);
        if (__fu__ == nullptr) {
            return ENOTCONN;
        }
        rpc::i32 __ret__ = __fu__->get_error_code();
        if (__ret__ == 0) {
            __fu__->get_reply() >> *flag;
        }
        __fu__->release();
        return __ret__;
    }
    rpc::Future* async_fast_dot_prod(const point3& p1, const point3& p2, const rpc::FutureAttr& __fu_attr__ = rpc::FutureAttr()) {
        rpc::Future* __fu__ = __cl__->begin_request(DemoService::FAST_DOT_PROD, __fu_attr__);
        if (__fu__ != nullptr) {
            *__cl__ << p1;
            *__cl__ << p2;
        }
        __cl__->end_request();
        return __fu__;
    }
    rpc::i32 fast_dot_prod(const point3& p1, const point3& p2, double* v) {
        rpc::Future* __fu__ = this->async_fast_dot_prod(p1, p2);
        if (__fu__ == nullptr) {
            return ENOTCONN;
        }
        rpc::i32 __ret__ = __fu__->get_error_code();
        if (__ret__ == 0) {
            __fu__->get_reply() >> *v;
        }
        __fu__->release();
        return __ret__;
    }
    rpc::Future* async_fast_large_str_nop(const std::string& str, const rpc::FutureAttr& __fu_attr__ = rpc::FutureAttr()) {
        rpc::Future* __fu__ = __cl__->begin_request(DemoService::FAST_LARGE_STR_NOP, __fu_attr__);
        if (__fu__ != nullptr) {
            *__cl__ << str;
        }
        __cl__->end_request();
        return __fu__;
    }
    rpc::i32 fast_large_str_nop(const std::string& str) {
        rpc::Future* __fu__ = this->async_fast_large_str_nop(str);
        if (__fu__ == nullptr) {
            return ENOTCONN;
        }
        rpc::i32 __ret__ = __fu__->get_error_code();
        __fu__->release();
        return __ret__;
    }
    rpc::Future* async_fast_vec_len(const std::vector<std::vector<std::string>>& v, const rpc::FutureAttr& __fu_attr__ = rpc::FutureAttr()) {
        rpc::Future* __fu__ = __cl__->begin_request(DemoService::FAST_VEC_LEN, __fu_attr__);
        if (__fu__ != nullptr) {
            *__cl__ << v;
        }
        __cl__->end_request();
        return __fu__;
    }
    rpc::i32 fast_vec_len(const std::vector<std::vector<std::string>>& v, rpc::i32* len) {
        rpc::Future* __fu__ = this->async_fast_vec_len(v);
        if (__fu__ == nullptr) {
            return ENOTCONN;
        }
        rpc::i32 __ret__ = __fu__->get_error_code();
        if (__ret__ == 0) {
            __fu__->get_reply() >> *len;
        }
        __fu__->release();
        return __ret__;
    }
    rpc::Future* async_prime(const rpc::i32& n, const rpc::FutureAttr& __fu_attr__ = rpc::FutureAttr()) {
        rpc::Future* __fu__ = __cl__->begin_request(DemoService::PRIME, __fu_attr__);
        if (__fu__ != nullptr) {
            *__cl__ << n;
        }
        __cl__->end_request();
        return __fu__;
    }
    rpc::i32 prime(const rpc::i32& n, rpc::i32* flag) {
        rpc::Future* __fu__ = this->async_prime(n);
        if (__fu__ == nullptr) {
            return ENOTCONN;
        }
        rpc::i32 __ret__ = __fu__->get_error_code();
        if (__ret__ == 0) {
            __fu__->get_reply() >> *flag;
        }
        __fu__->release();
        return __ret__;
    }
    rpc::Future* async_dot_prod(const point3& p1, const point3& p2, const rpc::FutureAttr& __fu_attr__ = rpc::FutureAttr()) {
        rpc::Future* __fu__ = __cl__->begin_request(DemoService::DOT_PROD, __fu_attr__);
        if (__fu__ != nullptr) {
            *__cl__ << p1;
            *__cl__ << p2;
        }
        __cl__->end_request();
        return __fu__;
    }
    rpc::i32 dot_prod(const point3& p1, const point3& p2, double* v) {
        rpc::Future* __fu__ = this->async_dot_prod(p1, p2);
        if (__fu__ == nullptr) {
            return ENOTCONN;
        }
        rpc::i32 __ret__ = __fu__->get_error_code();
        if (__ret__ == 0) {
            __fu__->get_reply() >> *v;
        }
        __fu__->release();
        return __ret__;
    }
    rpc::Future* async_large_str_nop(const std::string& str, const rpc::FutureAttr& __fu_attr__ = rpc::FutureAttr()) {
        rpc::Future* __fu__ = __cl__->begin_request(DemoService::LARGE_STR_NOP, __fu_attr__);
        if (__fu__ != nullptr) {
            *__cl__ << str;
        }
        __cl__->end_request();
        return __fu__;
    }
    rpc::i32 large_str_nop(const std::string& str) {
        rpc::Future* __fu__ = this->async_large_str_nop(str);
        if (__fu__ == nullptr) {
            return ENOTCONN;
        }
        rpc::i32 __ret__ = __fu__->get_error_code();
        __fu__->release();
        return __ret__;
    }
};

class NullService: public rpc::Service {
public:
    enum {
        TEST = 0x39fa9426,
    };
    int reg_to(rpc::Server* svr) {
        int ret = 0;
        if ((ret = svr->reg(TEST, this, &NullService::__test__wrapper__)) != 0) {
            goto err;
        }
        return 0;
    err:
        svr->unreg(TEST);
        return ret;
    }
    // these RPC handler functions need to be implemented by user
    // for 'raw' handlers, remember to reply req, delete req, and sconn->release(); use sconn->run_async for heavy job
    virtual void test(const rpc::i32& n, const rpc::i32& arg1, rpc::i32* result);
private:
    void __test__wrapper__(rpc::Request* req, rpc::ServerConnection* sconn) {
        auto f = [=] {
            rpc::i32 in_0;
            req->m >> in_0;
            rpc::i32 in_1;
            req->m >> in_1;
            rpc::i32 out_0;
            this->test(in_0, in_1, &out_0);
            sconn->begin_reply(req);
            *sconn << out_0;
            sconn->end_reply();
            delete req;
            sconn->release();
        };
        sconn->run_async(f);
    }
};

class NullProxy {
protected:
    rpc::Client* __cl__;
public:
    NullProxy(rpc::Client* cl): __cl__(cl) { }
    rpc::Future* async_test(const rpc::i32& n, const rpc::i32& arg1, const rpc::FutureAttr& __fu_attr__ = rpc::FutureAttr()) {
        rpc::Future* __fu__ = __cl__->begin_request(NullService::TEST, __fu_attr__);
        if (__fu__ != nullptr) {
            *__cl__ << n;
            *__cl__ << arg1;
        }
        __cl__->end_request();
        return __fu__;
    }
    rpc::i32 test(const rpc::i32& n, const rpc::i32& arg1, rpc::i32* result) {
        rpc::Future* __fu__ = this->async_test(n, arg1);
        if (__fu__ == nullptr) {
            return ENOTCONN;
        }
        rpc::i32 __ret__ = __fu__->get_error_code();
        if (__ret__ == 0) {
            __fu__->get_reply() >> *result;
        }
        __fu__->release();
        return __ret__;
    }
};

} // namespace demo

// optional %%: marks begining of C++ code, will be copied to end of generated header

namespace demo {

inline void DemoService::fast_dot_prod(const point3& p1, const point3& p2, double* v) {
    *v = p1.x * p2.x + p1.y * p2.y + p1.z * p2.z;
}

inline void DemoService::fast_large_str_nop(const std::string& str) { }

inline void DemoService::large_str_nop(const std::string& str) { }

inline void DemoService::fast_vec_len(const std::vector<std::vector<std::string>>& v, rpc::i32* len) {
    *len = v.size();
}

} // namespace demo

