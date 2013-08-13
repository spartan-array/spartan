#include <string>
#include <sstream>

#include <sys/select.h>
#include <errno.h>
#include <string.h>
#include <sys/types.h>
#include <sys/socket.h>
#include <netdb.h>
#include <netinet/tcp.h>

#include "server.h"

using namespace std;

namespace rpc {

void ServerConnection::run_async(const std::function<void()>& f) {
    server_->threadpool_->run_async(f);
}

void ServerConnection::begin_reply(Request* req, i32 error_code /* =... */) {
    out_l_.lock();

    bmark_ = this->out_.set_bookmark(sizeof(i32)); // will write reply size later

    *this << req->xid;
    *this << error_code;
}

void ServerConnection::end_reply() {
    // set reply size in packet
    if (bmark_ != NULL) {
        i32 reply_size = out_.get_and_reset_write_cnt();
        out_.write_bookmark(bmark_, &reply_size);
        delete bmark_;
        bmark_ = NULL;
    }

    if (!out_.empty()) {
        server_->pollmgr_->update_mode(this, Pollable::READ | Pollable::WRITE);
    }

    out_l_.unlock();
}

void ServerConnection::handle_read() {
    if (status_ == CLOSED) {
        return;
    }

    int bytes_read = in_.read_from_fd(socket_);
    if (bytes_read == 0) {
        return;
    }

    list<Request*> complete_requests;

    for (;;) {
        i32 packet_size;
        int n_peek = in_.peek(&packet_size, sizeof(i32));
        if (n_peek == sizeof(i32) && in_.content_size_gt(packet_size + sizeof(i32) - 1)) {
            // consume the packet size
            verify(in_.read(&packet_size, sizeof(i32)) == sizeof(i32));

            Request* req = new Request;
            verify(req->m.read_from_marshal(in_, packet_size) == (size_t) packet_size);

            if (packet_size < (int) sizeof(i64)) {
                Log::warn("rpc::ServerConnection: got an incomplete packet, xid not included");

                // Since we don't have xid, we don't know how to notify client about the failure.
                // All we can do is simply cleanup resource.
                delete req;
            } else {
                req->m >> req->xid;
                complete_requests.push_back(req);
            }

        } else {
            // packet not complete or there's no more packet to process
            break;
        }
    }

    for (list<Request*>::iterator iter = complete_requests.begin(); iter != complete_requests.end(); ++iter) {
        Request* req = *iter;

        if (!req->m.content_size_gt(sizeof(i32) - 1)) {
            // rpc id not provided
            begin_reply(req, EINVAL);
            end_reply();
            delete req;
            continue;
        }

        i32 rpc_id;
        req->m >> rpc_id;

        unordered_map<i32, Server::Handler*>::iterator it = server_->handlers_.find(rpc_id);
        if (it != server_->handlers_.end()) {
            // the handler should delete req, and release server_connection refcopy.
            it->second->handle(req, (ServerConnection *) this->ref_copy());
        } else {
            Log::error("rpc::ServerConnection: no handler for rpc_id=%d", rpc_id);
            begin_reply(req, ENOENT);
            end_reply();
            delete req;
        }
    }
}

void ServerConnection::handle_write(const io_ratelimit& rate) {
    if (status_ == CLOSED) {
        return;
    }

    out_l_.lock();
    Marshal::read_barrier barrier = out_.get_read_barrier();
    out_l_.unlock();

    out_.write_to_fd(socket_, barrier, rate);

    out_l_.lock();
    if (out_.empty()) {
        server_->pollmgr_->update_mode(this, Pollable::READ);
    }
    out_l_.unlock();
}

void ServerConnection::handle_error() {
    this->close();
}

void ServerConnection::close() {
    bool should_release = false;

    if (status_ == CONNECTED) {
        server_->sconns_l_.lock();
        unordered_set<ServerConnection*>::iterator it = server_->sconns_.find(this);
        if (it == server_->sconns_.end()) {
            // another thread has already calling close()
            server_->sconns_l_.unlock();
            return;
        }
        server_->sconns_.erase(it);

        // because we released this connection from server_->sconns_
        should_release = true;

        server_->pollmgr_->remove(this);
        server_->sconns_l_.unlock();

        Log::debug("rpc::ServerConnection: closed on fd=%d", socket_);

        status_ = CLOSED;
        ::close(socket_);
    }

    // this call might actually DELETE this object, so we put it to the end of function
    if (should_release) {
        this->release();
    }
}

int ServerConnection::poll_mode() {
    int mode = Pollable::READ;
    out_l_.lock();
    if (!out_.empty()) {
        mode |= Pollable::WRITE;
    }
    out_l_.unlock();
    return mode;
}

Server::Server(PollMgr* pollmgr /* =... */, ThreadPool* thrpool /* =? */)
        : server_sock_(-1), status_(NEW) {

    // get rid of eclipse warning
    memset(&loop_th_, 0, sizeof(loop_th_));

    if (pollmgr == NULL) {
        poll_options opt;
        opt.n_threads = 8;
        pollmgr_ = new PollMgr(opt);
    } else {
        pollmgr_ = (PollMgr *) pollmgr->ref_copy();
    }

    if (thrpool == NULL) {
        threadpool_ = new ThreadPool;
    } else {
        threadpool_ = (ThreadPool *) thrpool->ref_copy();
    }
}

Server::~Server() {
    if (status_ == RUNNING) {
        status_ = STOPPING;
        // wait till accepting thread done
        Pthread_join(loop_th_, NULL);

        verify(server_sock_ == -1 && status_ == STOPPED);
    }

    sconns_l_.lock();
    vector<ServerConnection*> sconns(sconns_.begin(), sconns_.end());
    sconns_l_.unlock();

    for (vector<ServerConnection*>::iterator it = sconns.begin(); it != sconns.end(); ++it) {
        (*it)->close();
    }

    // always release ThreadPool before PollMgr, in case some running task want to access PollMgr
    threadpool_->release();
    pollmgr_->release();

    for (auto& it : handlers_) {
        delete it.second;
    }

    //Log::debug("rpc::Server: destroyed");
}

struct start_server_loop_args_type {
    Server* server;
    struct addrinfo* gai_result;
    struct addrinfo* svr_addr;
};

void* Server::start_server_loop(void* arg) {
    start_server_loop_args_type* start_server_loop_args = (start_server_loop_args_type*) arg;

    start_server_loop_args->server->server_loop(start_server_loop_args->svr_addr);

    freeaddrinfo(start_server_loop_args->gai_result);
    delete start_server_loop_args;

    pthread_exit(NULL);
    return NULL;
}

void Server::server_loop(struct addrinfo* svr_addr) {
    fd_set fds;
    while (status_ == RUNNING) {
        FD_ZERO(&fds);
        FD_SET(server_sock_, &fds);

        // use select to avoid waiting on accept when closing server
        timeval tv;
        tv.tv_sec = 0;
        tv.tv_usec = 50 * 1000; // 0.05 sec
        int fdmax = server_sock_;

        int n_ready = select(fdmax + 1, &fds, NULL, NULL, &tv);
        if (n_ready == 0) {
            continue;
        }
        if (status_ != RUNNING) {
            break;
        }

        int clnt_socket = accept(server_sock_, svr_addr->ai_addr, &svr_addr->ai_addrlen);
        if (clnt_socket >= 0 && status_ == RUNNING) {
            Log::debug("rpc::Server: got new client, fd=%d", clnt_socket);
            verify(set_nonblocking(clnt_socket, true) == 0);

            sconns_l_.lock();
            ServerConnection* sconn = new ServerConnection(this, clnt_socket);
            sconns_.insert(sconn);
            pollmgr_->add(sconn);
            sconns_l_.unlock();
        }
    }

    close(server_sock_);
    server_sock_ = -1;
    status_ = STOPPED;
}

int Server::start(const char* bind_addr) {
    string addr(bind_addr);
    size_t idx = addr.find(":");
    if (idx == string::npos) {
        Log::error("rpc::Server: bad bind address: %s", bind_addr);
        errno = EINVAL;
        return -1;
    }
    string host = addr.substr(0, idx);
    string port = addr.substr(idx + 1);

    struct addrinfo hints, *result, *rp;
    memset(&hints, 0, sizeof(struct addrinfo));

    hints.ai_family = AF_INET; // ipv4
    hints.ai_socktype = SOCK_STREAM; // tcp
    hints.ai_flags = AI_PASSIVE; // server side

    int r = getaddrinfo((host == "0.0.0.0") ? NULL : host.c_str(), port.c_str(), &hints, &result);
    if (r != 0) {
        Log::error("rpc::Server: getaddrinfo(): %s", gai_strerror(r));
        return -1;
    }

    for (rp = result; rp != NULL; rp = rp->ai_next) {
        server_sock_ = socket(rp->ai_family, rp->ai_socktype, rp->ai_protocol);
        if (server_sock_ == -1) {
            continue;
        }

        const int yes = 1;
        verify(setsockopt(server_sock_, SOL_SOCKET, SO_REUSEADDR, &yes, sizeof(yes)) == 0);
        verify(setsockopt(server_sock_, IPPROTO_TCP, TCP_NODELAY, &yes, sizeof(yes)) == 0);

        if (::bind(server_sock_, rp->ai_addr, rp->ai_addrlen) == 0) {
            break;
        }
        close(server_sock_);
        server_sock_ = -1;
    }

    if (rp == NULL) {
        // failed to bind
        Log::error("rpc::Server: bind(): %s", strerror(errno));
        freeaddrinfo(result);
        return -1;
    }

    // about backlog: http://www.linuxjournal.com/files/linuxjournal.com/linuxjournal/articles/023/2333/2333s2.html
    const int backlog = SOMAXCONN;
    verify(listen(server_sock_, backlog) == 0);
    verify(set_nonblocking(server_sock_, true) == 0);

    status_ = RUNNING;
    Log::info("rpc::Server: started on %s", bind_addr);

    start_server_loop_args_type* start_server_loop_args = new start_server_loop_args_type();
    start_server_loop_args->server = this;
    start_server_loop_args->gai_result = result;
    start_server_loop_args->svr_addr = rp;
    Pthread_create(&loop_th_, NULL, Server::start_server_loop, start_server_loop_args);

    return 0;
}

void Server::reg(i32 rpc_id, void (*svc_func)(Request*, ServerConnection*)) {
    // disallow duplicate rpc_id
    verify(handlers_.find(rpc_id) == handlers_.end());

    class H: public Handler {
        void (*svc_func_)(Request*, ServerConnection*);
    public:
        H(void (*svc_func)(Request*, ServerConnection*))
                : svc_func_(svc_func) {
        }
        void handle(Request* req, ServerConnection* sconn) {
            svc_func_(req, sconn);
        }
    };

    handlers_[rpc_id] = new H(svc_func);
}

}
