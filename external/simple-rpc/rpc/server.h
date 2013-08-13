#pragma once

#include <unordered_map>
#include <unordered_set>

#include <pthread.h>

#include "marshal.h"
#include "polling.h"

#ifndef RPC_SERVER_H_
#define RPC_SERVER_H_
#endif // RPC_SERVER_H_

// for getaddrinfo() used in Server::start()
struct addrinfo;

namespace rpc {

class Server;

/**
 * The raw packet sent from client will be like this:
 * <size> <xid> <rpc_id> <arg1> <arg2> ... <argN>
 * NOTE: size does not include the size itself (<xid>..<argN>).
 *
 * For the request object, the marshal only contains <arg1>..<argN>,
 * other fields are already consumed.
 */
struct Request {
    Marshal m;
    i64 xid;
};

class Service {
public:
    virtual ~Service() {
    }
    virtual void reg_to(Server*) = 0;
};

class ServerConnection: public Pollable {

    friend class Server;

    Marshal in_, out_;
    ShortLock out_l_;

    Server* server_;
    int socket_;

    Marshal::bookmark* bmark_;

    enum {
        CONNECTED, CLOSED
    } status_;

    /**
     * Only to be called by:
     * 1: ~Server(), which is called when destroying Server
     * 2: handle_error(), which is called by PollMgr
     */
    void close();

protected:

    // Protected destructor as required by RefCounted.
    ~ServerConnection() {
        //Log::debug("rpc::ServerConnection: destroyed");
    }

public:

    ServerConnection(Server* server, int socket)
            : server_(server), socket_(socket), bmark_(NULL), status_(CONNECTED) { }

    /**
     * Start a reply message. Must be paired with end_reply().
     *
     * Reply message format:
     * <size> <xid> <error_code> <ret1> <ret2> ... <retN>
     * NOTE: size does not include size itself (<xid>..<retN>).
     *
     * User only need to fill <ret1>..<retN>.
     *
     * Currently used errno:
     * 0: everything is fine
     * ENOENT: method not found
     * EINVAL: invalid packet (field missing)
     */
    void begin_reply(Request* req, i32 error_code = 0);

    void end_reply();

    // helper function, do some work in background
    void run_async(const std::function<void()>& f);

    template<class T>
    ServerConnection& operator <<(const T& v) {
        this->out_ << v;
        return *this;
    }

    int fd() {
        return socket_;
    }

    int poll_mode();
    void handle_write(const io_ratelimit& rate);
    void handle_read();
    void handle_error();
};

class Server: public NoCopy {

    friend class ServerConnection;

    class Handler: public NoCopy {
    public:
        virtual void handle(Request* req, ServerConnection* sconn) = 0;
    };

    std::unordered_map<i32, Handler*> handlers_;
    PollMgr* pollmgr_;
    ThreadPool* threadpool_;
    int server_sock_;

    ShortLock sconns_l_;
    std::unordered_set<ServerConnection*> sconns_;

    enum {
        NEW, RUNNING, STOPPING, STOPPED
    } status_;

    pthread_t loop_th_;

    static void* start_server_loop(void* arg);
    void server_loop(struct addrinfo* svr_addr);

public:

    Server(PollMgr* pollmgr = NULL, ThreadPool* thrpool = NULL);
    virtual ~Server();

    int start(const char* bind_addr);

    void reg(Service* svc) {
        svc->reg_to(this);
    }

    /**
     * The svc_func need to do this:
     *
     *  {
     *     // process request
     *     ..
     *
     *     // send reply
     *     server_connection->begin_packet();
     *     *server_connection << {packet_content};
     *     server_connection->end_packet();
     *
     *     // cleanup resource
     *     delete request;
     *     server_connection->release();
     *  }
     */
    void reg(i32 rpc_id, void (*svc_func)(Request*, ServerConnection*));

    /**
     * The svc_func need to do this:
     *
     *  {
     *     // process request
     *     ..
     *
     *     // send reply
     *     server_connection->begin_packet();
     *     *server_connection << {packet_content};
     *     server_connection->end_packet();
     *
     *     // cleanup resource
     *     delete request;
     *     server_connection->release();
     *  }
     */
    template<class S>
    void reg(i32 rpc_id, S* svc, void (S::*svc_func)(Request*, ServerConnection*)) {

        // disallow duplicate rpc_id
        verify(handlers_.find(rpc_id) == handlers_.end());

        class H: public Handler {
            S* svc_;
            void (S::*svc_func_)(Request*, ServerConnection*);
        public:
            H(S* svc, void (S::*svc_func)(Request*, ServerConnection*))
                    : svc_(svc), svc_func_(svc_func) {
            }
            void handle(Request* req, ServerConnection* sconn) {
                (svc_->*svc_func_)(req, sconn);
            }
        };

        handlers_[rpc_id] = new H(svc, svc_func);
    }
};

}

