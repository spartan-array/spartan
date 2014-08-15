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


#ifdef RPC_STATISTICS

static const int g_stat_server_batching_size = 1000;
static int g_stat_server_batching[g_stat_server_batching_size];
static int g_stat_server_batching_idx;
static uint64_t g_stat_server_batching_report_time = 0;
static const uint64_t g_stat_server_batching_report_interval = 1000 * 1000 * 1000;

static void stat_server_batching(size_t batch) {
    g_stat_server_batching_idx = (g_stat_server_batching_idx + 1) % g_stat_server_batching_size;
    g_stat_server_batching[g_stat_server_batching_idx] = batch;
    uint64_t now = base::rdtsc();
    if (now - g_stat_server_batching_report_time > g_stat_server_batching_report_interval) {
        // do report
        int min = numeric_limits<int>::max();
        int max = 0;
        int sum_count = 0;
        int sum = 0;
        for (int i = 0; i < g_stat_server_batching_size; i++) {
            if (g_stat_server_batching[i] == 0) {
                continue;
            }
            if (g_stat_server_batching[i] > max) {
                max = g_stat_server_batching[i];
            }
            if (g_stat_server_batching[i] < min) {
                min = g_stat_server_batching[i];
            }
            sum += g_stat_server_batching[i];
            sum_count++;
            g_stat_server_batching[i] = 0;
        }
        double avg = double(sum) / sum_count;
        Log::info("* SERVER BATCHING: min=%d avg=%.1lf max=%d", min, avg, max);
        g_stat_server_batching_report_time = now;
    }
}

// rpc_id -> <count, cumulative>
static unordered_map<i32, pair<Counter, Counter>> g_stat_rpc_counter;
static uint64_t g_stat_server_rpc_counting_report_time = 0;
static const uint64_t g_stat_server_rpc_counting_report_interval = 1000 * 1000 * 1000;

static void stat_server_rpc_counting(i32 rpc_id) {
    g_stat_rpc_counter[rpc_id].first.next();

    uint64_t now = base::rdtsc();
    if (now - g_stat_server_rpc_counting_report_time > g_stat_server_rpc_counting_report_interval) {
        // do report
        for (auto& it: g_stat_rpc_counter) {
            i32 counted_rpc_id = it.first;
            i64 count = it.second.first.peek_next();
            it.second.first.reset();
            it.second.second.next(count);
            i64 cumulative = it.second.second.peek_next();
            Log::info("* RPC COUNT: id=%#08x count=%ld cumulative=%ld", counted_rpc_id, count, cumulative);
        }
        g_stat_server_rpc_counting_report_time = now;
    }
}

#endif // RPC_STATISTICS



int ServerConnection::run_async(const std::function<void()>& f, int queuing_channel /* =? */) {
    return server_->threadpool_->run_async(f, queuing_channel);
}



class ServerUdpConnection: public ServerConnection {
    char* udp_buffer_;

    virtual void close() {
        // will not be called
        verify(0);
    }

    virtual Marshal* output_buffer() {
        // will not be called
        verify(0);
        return nullptr;
    }

public:
    ServerUdpConnection(Server* svr, int udp_sock): ServerConnection(svr, udp_sock) {
        udp_buffer_ = new char[UdpBuffer::max_udp_packet_size_s];
    }
    ~ServerUdpConnection() {
        delete[] udp_buffer_;
    }
    virtual int poll_mode() {
        return Pollable::READ;  // always read only
    }
    void handle_write() {
        // will not be called
        verify(0);
    }
    void handle_read();
    void handle_error() {
        ::close(sock_);
    }

    virtual void begin_reply(Request* req, i32 error_code = 0) {
        // no reply, should not be called
        verify(0);
    }

    virtual void end_reply() {
        // no reply, should not be called
        verify(0);
    }
};


// <size> <rpc_id> <arg1> <arg2> ... <argN>
void ServerUdpConnection::handle_read() {
    int cnt = recvfrom(sock_, udp_buffer_, UdpBuffer::max_udp_packet_size_s, MSG_WAITALL, nullptr, nullptr);
    Marshal m_in;
    m_in.write(udp_buffer_, cnt);

    list<Request*> complete_requests;

    for (;;) {
        i32 packet_size;
        int n_peek = m_in.peek(&packet_size, sizeof(i32));
        if (n_peek == sizeof(i32) && m_in.content_size() >= packet_size + sizeof(i32)) {
            // consume the packet size
            verify(m_in.read(&packet_size, sizeof(i32)) == sizeof(i32));

            Request* req = new Request;
            verify(req->m.read_from_marshal(m_in, packet_size) == (size_t) packet_size);

            req->xid = -1;  // UDP packets does not have valid xid
            complete_requests.push_back(req);

        } else {
            // packet not complete or there's no more packet to process
            break;
        }
    }

#ifdef RPC_STATISTICS
    stat_server_batching(complete_requests.size());
#endif // RPC_STATISTICS

    for (auto& req: complete_requests) {

        if (req->m.content_size() < sizeof(i32)) {
            // rpc id not provided, discard
            delete req;
            continue;
        }

        i32 rpc_id;
        req->m >> rpc_id;

#ifdef RPC_STATISTICS
        stat_server_rpc_counting(rpc_id);
#endif // RPC_STATISTICS

        auto it = server_->handlers_.find(rpc_id);
        if (it != server_->handlers_.end()) {
            // the handler should delete req, and release server_connection refcopy.
            it->second(req, (ServerUdpConnection *) this->ref_copy());
        } else {
            Log_error("rpc::ServerConnection: no handler for rpc_id=0x%08x", rpc_id);
            delete req;
        }
    }
}



class ServerTcpConnection: public ServerConnection {

    friend class Server;

    Marshal in_, out_;
    SpinLock out_l_;

    bookmark* bmark_;

    enum {
        CONNECTED, CLOSED
    } status_;


    virtual Marshal* output_buffer() {
        return &out_;
    }

    /**
     * Only to be called by:
     * 1: ~Server(), which is called when destroying Server
     * 2: handle_error(), which is called by PollMgr
     */
    void close();

    // used to surpress multiple "no handler for rpc_id=..." errro
    static std::unordered_set<i32> rpc_id_missing_s;
    static SpinLock rpc_id_missing_l_s;

protected:

    // Protected destructor as required by RefCounted.
    ~ServerTcpConnection();

public:

    ServerTcpConnection(Server* server, int socket);

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

    int poll_mode();
    void handle_write();
    void handle_read();
    void handle_error();
};


std::unordered_set<i32> ServerTcpConnection::rpc_id_missing_s;
SpinLock ServerTcpConnection::rpc_id_missing_l_s;


ServerTcpConnection::ServerTcpConnection(Server* server, int socket)
        : ServerConnection(server, socket), bmark_(nullptr), status_(CONNECTED) {
    // increase number of open connections
    server_->sconns_ctr_.next(1);
}

ServerTcpConnection::~ServerTcpConnection() {
    // decrease number of open connections
    server_->sconns_ctr_.next(-1);
}


void ServerTcpConnection::begin_reply(Request* req, i32 error_code /* =... */) {
    out_l_.lock();
    v32 v_error_code = error_code;
    v64 v_reply_xid = req->xid;

    bmark_ = this->out_.set_bookmark(sizeof(i32)); // will write reply size later

    *this << v_reply_xid;
    *this << v_error_code;
}

void ServerTcpConnection::end_reply() {
    // set reply size in packet
    if (bmark_ != nullptr) {
        i32 reply_size = out_.get_and_reset_write_cnt();
        out_.write_bookmark(bmark_, &reply_size);
        delete bmark_;
        bmark_ = nullptr;
    }

    // always enable write events since the code above gauranteed there
    // will be some data to send
    server_->pollmgr_->update_mode(this, Pollable::READ | Pollable::WRITE);

    out_l_.unlock();
}

void ServerTcpConnection::handle_read() {
    if (status_ == CLOSED) {
        return;
    }

    int bytes_read = in_.read_from_fd(sock_);
    if (bytes_read == 0) {
        return;
    }

    list<Request*> complete_requests;

    for (;;) {
        i32 packet_size;
        int n_peek = in_.peek(&packet_size, sizeof(i32));
        if (n_peek == sizeof(i32) && in_.content_size() >= packet_size + sizeof(i32)) {
            // consume the packet size
            verify(in_.read(&packet_size, sizeof(i32)) == sizeof(i32));

            Request* req = new Request;
            verify(req->m.read_from_marshal(in_, packet_size) == (size_t) packet_size);

            v64 v_xid;
            req->m >> v_xid;
            req->xid = v_xid.get();
            complete_requests.push_back(req);

        } else {
            // packet not complete or there's no more packet to process
            break;
        }
    }

#ifdef RPC_STATISTICS
    stat_server_batching(complete_requests.size());
#endif // RPC_STATISTICS

    for (auto& req: complete_requests) {

        if (req->m.content_size() < sizeof(i32)) {
            // rpc id not provided
            begin_reply(req, EINVAL);
            end_reply();
            delete req;
            continue;
        }

        i32 rpc_id;
        req->m >> rpc_id;

#ifdef RPC_STATISTICS
        stat_server_rpc_counting(rpc_id);
#endif // RPC_STATISTICS

        auto it = server_->handlers_.find(rpc_id);
        if (it != server_->handlers_.end()) {
            // the handler should delete req, and release server_connection refcopy.
            it->second(req, (ServerConnection *) this->ref_copy());
        } else {
            rpc_id_missing_l_s.lock();
            bool surpress_warning = false;
            if (rpc_id_missing_s.find(rpc_id) == rpc_id_missing_s.end()) {
                rpc_id_missing_s.insert(rpc_id);
            } else {
                surpress_warning = true;
            }
            rpc_id_missing_l_s.unlock();
            if (!surpress_warning) {
                Log_error("rpc::ServerConnection: no handler for rpc_id=0x%08x", rpc_id);
            }
            begin_reply(req, ENOENT);
            end_reply();
            delete req;
        }
    }
}

void ServerTcpConnection::handle_write() {
    if (status_ == CLOSED) {
        return;
    }

    out_l_.lock();
    out_.write_to_fd(sock_);
    if (out_.empty()) {
        server_->pollmgr_->update_mode(this, Pollable::READ);
    }
    out_l_.unlock();
}

void ServerTcpConnection::handle_error() {
    this->close();
}

void ServerTcpConnection::close() {
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

        Log_debug("rpc::ServerConnection: closed on fd=%d", sock_);

        status_ = CLOSED;
        ::close(sock_);
    }

    // this call might actually DELETE this object, so we put it at the end of function
    if (should_release) {
        this->release();
    }
}

int ServerTcpConnection::poll_mode() {
    int mode = Pollable::READ;
    out_l_.lock();
    if (!out_.empty()) {
        mode |= Pollable::WRITE;
    }
    out_l_.unlock();
    return mode;
}

Server::Server(PollMgr* pollmgr /* =... */, ThreadPool* thrpool /* =? */)
        : server_sock_(-1), udp_(false), udp_sock_(-1), udp_conn_(nullptr), status_(NEW) {

    // get rid of eclipse warning
    memset(&loop_th_, 0, sizeof(loop_th_));

    if (pollmgr == nullptr) {
        pollmgr_ = new PollMgr;
    } else {
        pollmgr_ = (PollMgr *) pollmgr->ref_copy();
    }

    if (thrpool == nullptr) {
        threadpool_ = new ThreadPool;
    } else {
        threadpool_ = (ThreadPool *) thrpool->ref_copy();
    }
}

Server::~Server() {
    if (status_ == RUNNING) {
        status_ = STOPPING;
        // wait till accepting thread done
        Pthread_join(loop_th_, nullptr);

        verify(server_sock_ == -1 && status_ == STOPPED);
    }

    sconns_l_.lock();
    vector<ServerConnection*> sconns(sconns_.begin(), sconns_.end());
    // NOTE: do NOT clear sconns_ here, because when running the following
    // it->close(), the ServerConnection object will check the sconns_ to
    // ensure it still resides in sconns_
    sconns_l_.unlock();

    for (auto& it: sconns) {
        it->close();
    }

    if (udp_) {
        pollmgr_->remove(udp_conn_);
        udp_conn_->release();
    }

    // make sure all open connections are closed
    int alive_connection_count = -1;
    for (;;) {
        int new_alive_connection_count = sconns_ctr_.peek_next();
        if (new_alive_connection_count <= 0) {
            break;
        }
        if (alive_connection_count == -1 || new_alive_connection_count < alive_connection_count) {
            Log_debug("waiting for %d alive connections to shutdown", new_alive_connection_count);
        }
        alive_connection_count = new_alive_connection_count;
        // sleep 0.05 sec because this is the timeout for PollMgr's epoll()
        usleep(50 * 1000);
    }
    verify(sconns_ctr_.peek_next() == 0);

    threadpool_->release();
    pollmgr_->release();

    //Log_debug("rpc::Server: destroyed");
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

    pthread_exit(nullptr);
    return nullptr;
}

void Server::server_loop(struct addrinfo* svr_addr) {
    if (udp_) {
        udp_conn_ = new ServerUdpConnection(this, udp_sock_);
        pollmgr_->add(udp_conn_);
    }

    fd_set fds;
    while (status_ == RUNNING) {
        FD_ZERO(&fds);
        FD_SET(server_sock_, &fds);

        // use select to avoid waiting on accept when closing server
        timeval tv;
        tv.tv_sec = 0;
        tv.tv_usec = 50 * 1000; // 0.05 sec
        int fdmax = server_sock_;

        int n_ready = select(fdmax + 1, &fds, nullptr, nullptr, &tv);
        if (n_ready == 0) {
            continue;
        }
        if (status_ != RUNNING) {
            break;
        }

        int clnt_socket = accept(server_sock_, svr_addr->ai_addr, &svr_addr->ai_addrlen);
        if (clnt_socket >= 0 && status_ == RUNNING) {
            Log_debug("rpc::Server: got new client, fd=%d", clnt_socket);
            verify(set_nonblocking(clnt_socket, true) == 0);

            sconns_l_.lock();
            ServerConnection* sconn = new ServerTcpConnection(this, clnt_socket);
            sconns_.insert(sconn);
            pollmgr_->add(sconn);
            sconns_l_.unlock();
        }
    }

    close(server_sock_);
    server_sock_ = -1;
    if (udp_) {
        close(udp_sock_);
        udp_sock_ = -1;
    }
    status_ = STOPPED;
}

int Server::start(const char* bind_addr) {
    string addr(bind_addr);
    size_t idx = addr.find(":");
    if (idx == string::npos) {
        Log_error("rpc::Server: bad bind address: %s", bind_addr);
        return EINVAL;
    }
    string host = addr.substr(0, idx);
    string port = addr.substr(idx + 1);

    struct addrinfo hints, *result, *rp;
    memset(&hints, 0, sizeof(struct addrinfo));

    hints.ai_family = AF_INET;
    hints.ai_socktype = SOCK_STREAM; // tcp
    hints.ai_flags = AI_PASSIVE; // server side

    int r = getaddrinfo((host == "0.0.0.0") ? nullptr : host.c_str(), port.c_str(), &hints, &result);
    if (r != 0) {
        Log_error("rpc::Server: getaddrinfo(): %s", gai_strerror(r));
        return EINVAL;
    }

    for (rp = result; rp != nullptr; rp = rp->ai_next) {
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

    if (rp == nullptr) {
        // failed to bind
        Log_error("rpc::Server: bind(): %s", strerror(errno));
        freeaddrinfo(result);
        return EINVAL;
    }

    // about backlog: http://www.linuxjournal.com/files/linuxjournal.com/linuxjournal/articles/023/2333/2333s2.html
    const int backlog = SOMAXCONN;
    verify(listen(server_sock_, backlog) == 0);
    verify(set_nonblocking(server_sock_, true) == 0);

    if (udp_) {
        udp_sock_ = udp_bind(bind_addr);
        if (udp_sock_ == -1) {
            // failed to bind
            Log_error("rpc::Server: bind(): %s (UDP)", strerror(errno));
            // close the TCP socket opened
            close(server_sock_);
            server_sock_ = -1;
            return EINVAL;
        } else {
            Log_info("rpc::Server: started on %s (UDP)", bind_addr);
        }
    }

    status_ = RUNNING;
    Log_info("rpc::Server: started on %s", bind_addr);

    start_server_loop_args_type* start_server_loop_args = new start_server_loop_args_type();
    start_server_loop_args->server = this;
    start_server_loop_args->gai_result = result;
    start_server_loop_args->svr_addr = rp;
    Pthread_create(&loop_th_, nullptr, Server::start_server_loop, start_server_loop_args);

    return 0;
}

int Server::reg(i32 rpc_id, const std::function<void(Request*, ServerConnection*)>& func) {
    // disallow duplicate rpc_id
    if (handlers_.find(rpc_id) != handlers_.end()) {
        return EEXIST;
    }

    handlers_[rpc_id] = func;

    return 0;
}

void Server::unreg(i32 rpc_id) {
    handlers_.erase(rpc_id);
}

}
