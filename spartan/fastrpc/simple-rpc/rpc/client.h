#pragma once

#include <unordered_map>

#include "marshal.h"
#include "polling.h"

namespace rpc {

class Future;
class Client;

struct FutureAttr {
    FutureAttr(const std::function<void(Future*)>& cb = std::function<void(Future*)>()) : callback(cb) { }

    // callback should be fast, otherwise it hurts rpc performance
    std::function<void(Future*)> callback;
};

class Future: public RefCounted {
    friend class Client;

    i64 xid_;
    i32 error_code_;

    FutureAttr attr_;
    Marshal reply_;

    bool ready_;
    bool timed_out_;
    pthread_cond_t ready_cond_;
    pthread_mutex_t ready_m_;

    void notify_ready();

protected:

    // protected destructor as required by RefCounted.
    ~Future() {
        Pthread_mutex_destroy(&ready_m_);
        Pthread_cond_destroy(&ready_cond_);
    }

public:

    Future(i64 xid, const FutureAttr& attr = FutureAttr())
            : xid_(xid), error_code_(0), attr_(attr), ready_(false), timed_out_(false) {
        Pthread_mutex_init(&ready_m_, nullptr);
        Pthread_cond_init(&ready_cond_, nullptr);
    }

    bool ready() {
        Pthread_mutex_lock(&ready_m_);
        bool r = ready_;
        Pthread_mutex_unlock(&ready_m_);
        return r;
    }

    // wait till reply done
    void wait();

    void timed_wait(double sec);

    Marshal& get_reply() {
        wait();
        return reply_;
    }

    i32 get_error_code() {
        wait();
        return error_code_;
    }

    static inline void safe_release(Future* fu) {
        if (fu != nullptr) {
            fu->release();
        }
    }
};

class FutureGroup {
private:
    std::vector<Future*> futures_;

public:
    void add(Future* f) {
        if (f == nullptr) {
            Log_error("Invalid Future object passed to FutureGroup!");
            return;
        }
        futures_.push_back(f);
    }

    void wait_all() {
        for (auto& f : futures_) {
            f->wait();
        }
    }

    ~FutureGroup() {
        wait_all();
        for (auto& f : futures_) {
            f->release();
        }
    }
};

class Client: public Pollable {
    Marshal in_, out_;

    SpinLock udp_l_;
    int udp_sock_;
    UdpBuffer udp_;
    socklen_t udp_salen_;
    struct sockaddr *udp_sa_;
    bookmark* udp_bmark_;

    /**
     * NOT a refcopy! This is intended to avoid circular reference, which prevents everything from being released correctly.
     */
    PollMgr* pollmgr_;

    int sock_;
    enum {
        NEW, CONNECTED, CLOSED
    } status_;

    bookmark* bmark_;

    Counter xid_counter_;
    std::unordered_map<i64, Future*> pending_fu_;

    SpinLock pending_fu_l_;
    SpinLock out_l_;

    // reentrant, could be called multiple times before releasing
    void close();

    void invalidate_pending_futures();

    // prevent direct usage, use close_and_release() instead
    using RefCounted::release;

protected:

    virtual ~Client();

public:

    Client(PollMgr* pollmgr): udp_sock_(-1), udp_sa_(nullptr), udp_bmark_(nullptr), pollmgr_(pollmgr), sock_(-1), status_(NEW), bmark_(nullptr) { }

    /**
     * Start a new request. Must be paired with end_request(), even if nullptr returned.
     *
     * The request packet format is: <size> <xid> <rpc_id> <arg1> <arg2> ... <argN>
     */
    Future* begin_request(i32 rpc_id, const FutureAttr& attr = FutureAttr());

    void end_request();

    void begin_udp_request(i32 rpc_id);
    UdpBuffer& udp_request() {
        return udp_;
    }
    int end_udp_request();

    template<class T>
    Client& operator <<(const T& v) {
        if (status_ == CONNECTED) {
            this->out_ << v;
        }
        return *this;
    }

    // NOTE: this function is used *internally* by Python extension
    Client& operator <<(Marshal& m) {
        if (status_ == CONNECTED) {
            this->out_.read_from_marshal(m, m.content_size());
        }
        return *this;
    }

    int connect(const char* addr);

    void close_and_release() {
        close();
        release();
    }

    int fd() {
        return sock_;
    }

    int poll_mode();
    void handle_read();
    void handle_write();
    void handle_error();

};

class ClientPool: public NoCopy {
    rpc::Rand rand_;

    // refcopy
    rpc::PollMgr* pollmgr_;

    // guard cache_
    SpinLock l_;
    std::map<std::string, rpc::Client**> cache_;
    int parallel_connections_;

public:

    ClientPool(rpc::PollMgr* pollmgr = nullptr, int parallel_connections = 1);
    ~ClientPool();

    // return cached client connection
    // on error, return nullptr
    rpc::Client* get_client(const std::string& addr);

};

}
