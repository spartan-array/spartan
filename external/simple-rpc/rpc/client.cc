#include <string>

#include <errno.h>
#include <string.h>
#include <sys/types.h>
#include <sys/socket.h>
#include <netdb.h>
#include <netinet/tcp.h>

#include "client.h"

using namespace std;

namespace rpc {

void Future::wait() {
    Pthread_mutex_lock(&ready_m_);
    while (!ready_) {
        Pthread_cond_wait(&ready_cond_, &ready_m_);
    }
    Pthread_mutex_unlock(&ready_m_);
}

void Future::notify_ready() {
    Pthread_mutex_lock(&ready_m_);
    ready_ = true;
    Pthread_cond_signal(&ready_cond_);
    Pthread_mutex_unlock(&ready_m_);
    if (attr_.callback != nullptr) {
        attr_.callback(this);
    }
}

void Client::invalidate_pending_futures() {
    list<Future*> futures;
    pending_fu_l_.lock();
    while (pending_fu_.empty() == false) {
        futures.push_back(pending_fu_.begin()->second);
        pending_fu_.erase(pending_fu_.begin());
    }
    pending_fu_l_.unlock();

    for (list<Future*>::iterator it = futures.begin(); it != futures.end(); ++it) {
        Future* fu = *it;
        if (fu != NULL) {
            fu->error_code_ = ENOTCONN;
            fu->notify_ready();

            // since we removed it from pending_fu_
            fu->release();
        }
    }

}

void Client::close() {
    if (status_ == CONNECTED) {
        pollmgr_->remove(this);
        ::close(sock_);
        status_ = CLOSED;

        invalidate_pending_futures();
    }
    status_ = CLOSED;
}

int Client::connect(const char* addr) {
    string addr_str(addr);
    size_t idx = addr_str.find(":");
    if (idx == string::npos) {
        Log_error("rpc::Client: bad connect address: %s", addr);
        errno = EINVAL;
        return -1;
    }
    string host = addr_str.substr(0, idx);
    string port = addr_str.substr(idx + 1);

    struct addrinfo hints, *result, *rp;
    memset(&hints, 0, sizeof(struct addrinfo));

    hints.ai_family = AF_INET; // ipv4
    hints.ai_socktype = SOCK_STREAM; // tcp

    int r = getaddrinfo(host.c_str(), port.c_str(), &hints, &result);
    if (r != 0) {
        Log_error("rpc::Client: getaddrinfo(): %s", gai_strerror(r));
        return -1;
    }

    for (rp = result; rp != NULL; rp = rp->ai_next) {
        sock_ = socket(rp->ai_family, rp->ai_socktype, rp->ai_protocol);
        if (sock_ == -1) {
            continue;
        }

        const int yes = 1;
        verify(setsockopt(sock_, SOL_SOCKET, SO_REUSEADDR, &yes, sizeof(yes)) == 0);
        verify(setsockopt(sock_, IPPROTO_TCP, TCP_NODELAY, &yes, sizeof(yes)) == 0);

        if (::connect(sock_, rp->ai_addr, rp->ai_addrlen) == 0) {
            break;
        }
        ::close(sock_);
        sock_ = -1;
    }
    freeaddrinfo(result);

    if (rp == NULL) {
        // failed to connect
        Log_error("rpc::Client: connect(%s): %s", addr, strerror(errno));
        return -1;
    }

    verify(set_nonblocking(sock_, true) == 0);
    Log_debug("rpc::Client: connected to %s", addr);

    status_ = CONNECTED;
    pollmgr_->add(this);

    return 0;
}

void Client::handle_error() {
    close();
}

void Client::handle_write(const io_ratelimit& rate) {
    if (status_ != CONNECTED) {
        return;
    }

    out_l_.lock();
    Marshal::read_barrier barrier = out_.get_read_barrier();
//    out_l_.unlock();

    out_.write_to_fd(sock_, barrier, rate);

//    out_l_.lock();
    if (out_.empty()) {
        pollmgr_->update_mode(this, Pollable::READ);
    }
    out_l_.unlock();
}

void Client::handle_read() {
    if (status_ != CONNECTED) {
        return;
    }

    int bytes_read = in_.read_from_fd(sock_);
    if (bytes_read == 0) {
        return;
    }

    for (;;) {
        i32 packet_size;
        int n_peek = in_.peek(&packet_size, sizeof(i32));
        if (n_peek == sizeof(i32) && in_.content_size_gt(packet_size + sizeof(i32) - 1)) {
            // consume the packet size
            verify(in_.read(&packet_size, sizeof(i32)) == sizeof(i32));

            i64 reply_xid;
            i32 error_code;

            in_ >> reply_xid >> error_code;

            pending_fu_l_.lock();
            unordered_map<i64, Future*>::iterator it = pending_fu_.find(reply_xid);
            if (it != pending_fu_.end()) {
                Future* fu = it->second;
                verify(fu->xid_ == reply_xid);
                pending_fu_.erase(it);
                pending_fu_l_.unlock();

                fu->error_code_ = error_code;
                fu->reply_.read_from_marshal(in_, packet_size - sizeof(reply_xid) - sizeof(error_code));

                fu->notify_ready();

                // since we removed it from pending_fu_
                fu->release();
            } else {
                pending_fu_l_.unlock();
            }

        } else {
            // packet incomplete or no more packets to process
            break;
        }
    }
}

int Client::poll_mode() {
    int mode = Pollable::READ;
    out_l_.lock();
    if (!out_.empty()) {
        mode |= Pollable::WRITE;
    }
    out_l_.unlock();
    return mode;
}

Future* Client::begin_request(i32 rpc_id, const FutureAttr& attr /* =... */) {
    out_l_.lock();

    if (status_ != CONNECTED) {
        return NULL;
    }

    Future* fu = new Future(xid_counter_.next(), attr);
    pending_fu_l_.lock();
    pending_fu_[fu->xid_] = fu;
    pending_fu_.size();
    pending_fu_l_.unlock();

    // check if the client gets closed in the meantime
    if (status_ != CONNECTED) {
        pending_fu_l_.lock();
        unordered_map<i64, Future*>::iterator it = pending_fu_.find(fu->xid_);
        if (it != pending_fu_.end()) {
            it->second->release();
            pending_fu_.erase(it);
        }
        pending_fu_l_.unlock();

        return NULL;
    }

    bmark_ = out_.set_bookmark(sizeof(i32)); // will fill packet size later

    *this << fu->xid_;
    *this << rpc_id;

    // one ref is already in pending_fu_
    return (Future *) fu->ref_copy();
}

void Client::end_request() {
    // set reply size in packet
    if (bmark_ != NULL) {
        i32 request_size = out_.get_and_reset_write_cnt();
        out_.write_bookmark(bmark_, &request_size);
        out_.update_read_barrier();
        delete bmark_;
        bmark_ = NULL;
    }

    // always enable write events since the code above gauranteed there
    // will be some data to send
    pollmgr_->update_mode(this, Pollable::READ | Pollable::WRITE);

    out_l_.unlock();
}

ClientPool::ClientPool(PollMgr* pollmgr /* =? */, int parallel_connections /* =? */)
        : parallel_connections_(parallel_connections) {

    if (pollmgr == NULL) {
        pollmgr_ = new PollMgr;
    } else {
        pollmgr_ = (PollMgr *) pollmgr->ref_copy();
    }
}

ClientPool::~ClientPool() {
    for (auto& it : cache_) {
        for (int i = 0; i < parallel_connections_; i++) {
            it.second[i]->close_and_release();
        }
        delete[] it.second;
    }
    pollmgr_->release();
}

Client* ClientPool::get_client(const string& addr) {
    Client* cl = NULL;
    l_.lock();
    map<string, Client**>::iterator it = cache_.find(addr);
    if (it != cache_.end()) {
        cl = it->second[rand_() % parallel_connections_];
    } else {
        Client** parallel_clients = new Client*[parallel_connections_];
        int i;
        bool ok = true;
        for (i = 0; i < parallel_connections_; i++) {
            parallel_clients[i] = new Client(this->pollmgr_);
            if (parallel_clients[i]->connect(addr.c_str()) != 0) {
                ok = false;
                break;
            }
        }
        if (ok) {
            cl = parallel_clients[rand_() % parallel_connections_];
            cache_.insert(std::map<std::string, rpc::Client**>::value_type(addr, parallel_clients));
        } else {
            // close connections
            while (i >= 0) {
                parallel_clients[i]->close_and_release();
                i--;
            }
            delete[] parallel_clients;
        }
    }
    l_.unlock();
    return cl;
}

}
