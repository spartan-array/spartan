#pragma once

#include "rpc/server.h"
#include "rpc/client.h"

#include <errno.h>

// #include <math.h>

// optional %%: marks header section, code above will be copied into begin of generated C++ header
namespace benchmark {

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

class BenchmarkService: public rpc::Service {
public:
    enum {
        FAST_PRIME = 0x5b9365eb,
        FAST_DOT_PROD = 0x4a92c615,
        FAST_ADD = 0x484806bf,
        FAST_NOP = 0x4f29b7a5,
        PRIME = 0x1f96b90f,
        DOT_PROD = 0x3a2fd47a,
        ADD = 0x5135fabc,
        NOP = 0x3823ed06,
        SLEEP = 0x39d768e9,
        ADD_LATER = 0x64e0a594,
        LOSSY_NOP = 0x5323ea07,
        FAST_LOSSY_NOP = 0x6223e760,
    };
    int __reg_to__(rpc::Server* svr) {
        int ret = 0;
        if ((ret = svr->reg(FAST_PRIME, this, &BenchmarkService::__fast_prime__wrapper__)) != 0) {
            goto err;
        }
        if ((ret = svr->reg(FAST_DOT_PROD, this, &BenchmarkService::__fast_dot_prod__wrapper__)) != 0) {
            goto err;
        }
        if ((ret = svr->reg(FAST_ADD, this, &BenchmarkService::__fast_add__wrapper__)) != 0) {
            goto err;
        }
        if ((ret = svr->reg(FAST_NOP, this, &BenchmarkService::__fast_nop__wrapper__)) != 0) {
            goto err;
        }
        if ((ret = svr->reg(PRIME, this, &BenchmarkService::__prime__wrapper__)) != 0) {
            goto err;
        }
        if ((ret = svr->reg(DOT_PROD, this, &BenchmarkService::__dot_prod__wrapper__)) != 0) {
            goto err;
        }
        if ((ret = svr->reg(ADD, this, &BenchmarkService::__add__wrapper__)) != 0) {
            goto err;
        }
        if ((ret = svr->reg(NOP, this, &BenchmarkService::__nop__wrapper__)) != 0) {
            goto err;
        }
        if ((ret = svr->reg(SLEEP, this, &BenchmarkService::__sleep__wrapper__)) != 0) {
            goto err;
        }
        if ((ret = svr->reg(ADD_LATER, this, &BenchmarkService::__add_later__wrapper__)) != 0) {
            goto err;
        }
        svr->enable_udp();
        if ((ret = svr->reg(LOSSY_NOP, this, &BenchmarkService::__lossy_nop__wrapper__)) != 0) {
            goto err;
        }
        if ((ret = svr->reg(FAST_LOSSY_NOP, this, &BenchmarkService::__fast_lossy_nop__wrapper__)) != 0) {
            goto err;
        }
        return 0;
    err:
        svr->unreg(FAST_PRIME);
        svr->unreg(FAST_DOT_PROD);
        svr->unreg(FAST_ADD);
        svr->unreg(FAST_NOP);
        svr->unreg(PRIME);
        svr->unreg(DOT_PROD);
        svr->unreg(ADD);
        svr->unreg(NOP);
        svr->unreg(SLEEP);
        svr->unreg(ADD_LATER);
        svr->unreg(LOSSY_NOP);
        svr->unreg(FAST_LOSSY_NOP);
        return ret;
    }
    // these RPC handler functions need to be implemented by user
    // for 'raw' handlers, remember to reply req, delete req, and sconn->release(); use sconn->run_async for heavy job
    virtual void fast_prime(const rpc::i32& n, rpc::i8* flag);
    virtual void fast_dot_prod(const point3& p1, const point3& p2, double* v);
    virtual void fast_add(const rpc::v32& a, const rpc::v32& b, rpc::v32* a_add_b);
    virtual void fast_nop(const std::string&);
    virtual void prime(const rpc::i32& n, rpc::i8* flag);
    virtual void dot_prod(const point3& p1, const point3& p2, double* v);
    virtual void add(const rpc::v32& a, const rpc::v32& b, rpc::v32* a_add_b);
    virtual void nop(const std::string&);
    virtual void sleep(const double& sec);
    virtual void add_later(const rpc::i32& a, const rpc::i32& b, rpc::i32* sum, rpc::DeferredReply* defer);
    virtual void lossy_nop(const rpc::i32& dummy, const rpc::i32& dummy2);
    virtual void fast_lossy_nop();
private:
    void __fast_prime__wrapper__(rpc::Request* req, rpc::ServerConnection* sconn) {
        rpc::i32 in_0;
        req->m >> in_0;
        rpc::i8 out_0;
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
    void __fast_add__wrapper__(rpc::Request* req, rpc::ServerConnection* sconn) {
        rpc::v32 in_0;
        req->m >> in_0;
        rpc::v32 in_1;
        req->m >> in_1;
        rpc::v32 out_0;
        this->fast_add(in_0, in_1, &out_0);
        sconn->begin_reply(req);
        *sconn << out_0;
        sconn->end_reply();
        delete req;
        sconn->release();
    }
    void __fast_nop__wrapper__(rpc::Request* req, rpc::ServerConnection* sconn) {
        std::string in_0;
        req->m >> in_0;
        this->fast_nop(in_0);
        sconn->begin_reply(req);
        sconn->end_reply();
        delete req;
        sconn->release();
    }
    void __prime__wrapper__(rpc::Request* req, rpc::ServerConnection* sconn) {
        auto f = [=] {
            rpc::i32 in_0;
            req->m >> in_0;
            rpc::i8 out_0;
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
    void __add__wrapper__(rpc::Request* req, rpc::ServerConnection* sconn) {
        auto f = [=] {
            rpc::v32 in_0;
            req->m >> in_0;
            rpc::v32 in_1;
            req->m >> in_1;
            rpc::v32 out_0;
            this->add(in_0, in_1, &out_0);
            sconn->begin_reply(req);
            *sconn << out_0;
            sconn->end_reply();
            delete req;
            sconn->release();
        };
        sconn->run_async(f);
    }
    void __nop__wrapper__(rpc::Request* req, rpc::ServerConnection* sconn) {
        auto f = [=] {
            std::string in_0;
            req->m >> in_0;
            this->nop(in_0);
            sconn->begin_reply(req);
            sconn->end_reply();
            delete req;
            sconn->release();
        };
        sconn->run_async(f);
    }
    void __sleep__wrapper__(rpc::Request* req, rpc::ServerConnection* sconn) {
        auto f = [=] {
            double in_0;
            req->m >> in_0;
            this->sleep(in_0);
            sconn->begin_reply(req);
            sconn->end_reply();
            delete req;
            sconn->release();
        };
        sconn->run_async(f);
    }
    void __add_later__wrapper__(rpc::Request* req, rpc::ServerConnection* sconn) {
        rpc::i32* in_0 = new rpc::i32;
        req->m >> *in_0;
        rpc::i32* in_1 = new rpc::i32;
        req->m >> *in_1;
        rpc::i32* out_0 = new rpc::i32;
        auto __marshal_reply__ = [=] {
            *sconn << *out_0;
        };
        auto __cleanup__ = [=] {
            delete in_0;
            delete in_1;
            delete out_0;
        };
        rpc::DeferredReply* __defer__ = new rpc::DeferredReply(req, sconn, __marshal_reply__, __cleanup__);
        this->add_later(*in_0, *in_1, out_0, __defer__);
    }
    void __lossy_nop__wrapper__(rpc::Request* req, rpc::ServerConnection* sconn) {
        auto f = [=] {
            rpc::i32 in_0;
            req->m >> in_0;
            rpc::i32 in_1;
            req->m >> in_1;
            this->lossy_nop(in_0, in_1);
            delete req;
            sconn->release();
        };
        sconn->run_async(f);
    }
    void __fast_lossy_nop__wrapper__(rpc::Request* req, rpc::ServerConnection* sconn) {
        this->fast_lossy_nop();
        delete req;
        sconn->release();
    }
};

class BenchmarkProxy {
protected:
    rpc::Client* __cl__;
public:
    BenchmarkProxy(rpc::Client* cl): __cl__(cl) { }
    rpc::Future* async_fast_prime(const rpc::i32& n, const rpc::FutureAttr& __fu_attr__ = rpc::FutureAttr()) {
        rpc::Future* __fu__ = __cl__->begin_request(BenchmarkService::FAST_PRIME, __fu_attr__);
        if (__fu__ != nullptr) {
            *__cl__ << n;
        }
        __cl__->end_request();
        return __fu__;
    }
    rpc::i32 fast_prime(const rpc::i32& n, rpc::i8* flag) {
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
        rpc::Future* __fu__ = __cl__->begin_request(BenchmarkService::FAST_DOT_PROD, __fu_attr__);
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
    rpc::Future* async_fast_add(const rpc::v32& a, const rpc::v32& b, const rpc::FutureAttr& __fu_attr__ = rpc::FutureAttr()) {
        rpc::Future* __fu__ = __cl__->begin_request(BenchmarkService::FAST_ADD, __fu_attr__);
        if (__fu__ != nullptr) {
            *__cl__ << a;
            *__cl__ << b;
        }
        __cl__->end_request();
        return __fu__;
    }
    rpc::i32 fast_add(const rpc::v32& a, const rpc::v32& b, rpc::v32* a_add_b) {
        rpc::Future* __fu__ = this->async_fast_add(a, b);
        if (__fu__ == nullptr) {
            return ENOTCONN;
        }
        rpc::i32 __ret__ = __fu__->get_error_code();
        if (__ret__ == 0) {
            __fu__->get_reply() >> *a_add_b;
        }
        __fu__->release();
        return __ret__;
    }
    rpc::Future* async_fast_nop(const std::string& in_0, const rpc::FutureAttr& __fu_attr__ = rpc::FutureAttr()) {
        rpc::Future* __fu__ = __cl__->begin_request(BenchmarkService::FAST_NOP, __fu_attr__);
        if (__fu__ != nullptr) {
            *__cl__ << in_0;
        }
        __cl__->end_request();
        return __fu__;
    }
    rpc::i32 fast_nop(const std::string& in_0) {
        rpc::Future* __fu__ = this->async_fast_nop(in_0);
        if (__fu__ == nullptr) {
            return ENOTCONN;
        }
        rpc::i32 __ret__ = __fu__->get_error_code();
        __fu__->release();
        return __ret__;
    }
    rpc::Future* async_prime(const rpc::i32& n, const rpc::FutureAttr& __fu_attr__ = rpc::FutureAttr()) {
        rpc::Future* __fu__ = __cl__->begin_request(BenchmarkService::PRIME, __fu_attr__);
        if (__fu__ != nullptr) {
            *__cl__ << n;
        }
        __cl__->end_request();
        return __fu__;
    }
    rpc::i32 prime(const rpc::i32& n, rpc::i8* flag) {
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
        rpc::Future* __fu__ = __cl__->begin_request(BenchmarkService::DOT_PROD, __fu_attr__);
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
    rpc::Future* async_add(const rpc::v32& a, const rpc::v32& b, const rpc::FutureAttr& __fu_attr__ = rpc::FutureAttr()) {
        rpc::Future* __fu__ = __cl__->begin_request(BenchmarkService::ADD, __fu_attr__);
        if (__fu__ != nullptr) {
            *__cl__ << a;
            *__cl__ << b;
        }
        __cl__->end_request();
        return __fu__;
    }
    rpc::i32 add(const rpc::v32& a, const rpc::v32& b, rpc::v32* a_add_b) {
        rpc::Future* __fu__ = this->async_add(a, b);
        if (__fu__ == nullptr) {
            return ENOTCONN;
        }
        rpc::i32 __ret__ = __fu__->get_error_code();
        if (__ret__ == 0) {
            __fu__->get_reply() >> *a_add_b;
        }
        __fu__->release();
        return __ret__;
    }
    rpc::Future* async_nop(const std::string& in_0, const rpc::FutureAttr& __fu_attr__ = rpc::FutureAttr()) {
        rpc::Future* __fu__ = __cl__->begin_request(BenchmarkService::NOP, __fu_attr__);
        if (__fu__ != nullptr) {
            *__cl__ << in_0;
        }
        __cl__->end_request();
        return __fu__;
    }
    rpc::i32 nop(const std::string& in_0) {
        rpc::Future* __fu__ = this->async_nop(in_0);
        if (__fu__ == nullptr) {
            return ENOTCONN;
        }
        rpc::i32 __ret__ = __fu__->get_error_code();
        __fu__->release();
        return __ret__;
    }
    rpc::Future* async_sleep(const double& sec, const rpc::FutureAttr& __fu_attr__ = rpc::FutureAttr()) {
        rpc::Future* __fu__ = __cl__->begin_request(BenchmarkService::SLEEP, __fu_attr__);
        if (__fu__ != nullptr) {
            *__cl__ << sec;
        }
        __cl__->end_request();
        return __fu__;
    }
    rpc::i32 sleep(const double& sec) {
        rpc::Future* __fu__ = this->async_sleep(sec);
        if (__fu__ == nullptr) {
            return ENOTCONN;
        }
        rpc::i32 __ret__ = __fu__->get_error_code();
        __fu__->release();
        return __ret__;
    }
    rpc::Future* async_add_later(const rpc::i32& a, const rpc::i32& b, const rpc::FutureAttr& __fu_attr__ = rpc::FutureAttr()) {
        rpc::Future* __fu__ = __cl__->begin_request(BenchmarkService::ADD_LATER, __fu_attr__);
        if (__fu__ != nullptr) {
            *__cl__ << a;
            *__cl__ << b;
        }
        __cl__->end_request();
        return __fu__;
    }
    rpc::i32 add_later(const rpc::i32& a, const rpc::i32& b, rpc::i32* sum) {
        rpc::Future* __fu__ = this->async_add_later(a, b);
        if (__fu__ == nullptr) {
            return ENOTCONN;
        }
        rpc::i32 __ret__ = __fu__->get_error_code();
        if (__ret__ == 0) {
            __fu__->get_reply() >> *sum;
        }
        __fu__->release();
        return __ret__;
    }
    int lossy_nop(const rpc::i32& dummy, const rpc::i32& dummy2) /* UDP */ {
        __cl__->begin_udp_request(BenchmarkService::LOSSY_NOP);
        __cl__->udp_request() << dummy;
        __cl__->udp_request() << dummy2;
        return __cl__->end_udp_request();
    }
    int fast_lossy_nop() /* UDP */ {
        __cl__->begin_udp_request(BenchmarkService::FAST_LOSSY_NOP);
        return __cl__->end_udp_request();
    }
};

} // namespace benchmark


// optional %%: marks footer section, code below will be copied into end of generated C++ header

namespace benchmark {

inline void BenchmarkService::fast_dot_prod(const point3& p1, const point3& p2, double* v) {
    *v = p1.x * p2.x + p1.y * p2.y + p1.z * p2.z;
}

inline void BenchmarkService::fast_add(const rpc::v32& a, const rpc::v32& b, rpc::v32* a_add_b) {
    a_add_b->set(a.get() + b.get());
}

inline void BenchmarkService::prime(const rpc::i32& n, rpc::i8* flag) {
    return fast_prime(n, flag);
}

inline void BenchmarkService::dot_prod(const point3& p1, const point3& p2, double *v) {
    *v = p1.x * p2.x + p1.y * p2.y + p1.z * p2.z;
}

inline void BenchmarkService::add(const rpc::v32& a, const rpc::v32& b, rpc::v32* a_add_b) {
    a_add_b->set(a.get() + b.get());
}

inline void BenchmarkService::fast_nop(const std::string& str) {
    nop(str);
}

inline void BenchmarkService::lossy_nop(const rpc::i32& dummy, const rpc::i32& dummy2) {
    nop("");
}

inline void BenchmarkService::fast_lossy_nop() {
    nop("");
}

} // namespace benchmark

