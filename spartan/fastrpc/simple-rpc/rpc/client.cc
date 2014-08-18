#include <string>

#include <errno.h>

#include "client.h"

using namespace std;

namespace rpc {

void Future::wait() {
    Pthread_mutex_lock(&ready_m_);
    while (!ready_ && !timed_out_) {
        Pthread_cond_wait(&ready_cond_, &ready_m_);
    }
    Pthread_mutex_unlock(&ready_m_);
}

void Future::timed_wait(double sec) {
    Pthread_mutex_lock(&ready_m_);
    while (!ready_ && !timed_out_) {
        int full_sec = (int) sec;
        int nsec = int((sec - full_sec) * 1000 * 1000 * 1000);
        struct timeval tv;
        gettimeofday(&tv, nullptr);
        timespec abstime;
        abstime.tv_sec = tv.tv_sec + full_sec;
        abstime.tv_nsec = tv.tv_usec * 1000 + nsec;
        if (abstime.tv_nsec > 1000 * 1000 * 1000) {
            abstime.tv_nsec -= 1000 * 1000 * 1000;
            abstime.tv_sec += 1;
        }
        Log::debug("wait for %lf", sec);
        int ret = pthread_cond_timedwait(&ready_cond_, &ready_m_, &abstime);
        if (ret == ETIMEDOUT) {
            timed_out_ = true;
        } else {
            verify(ret == 0);
        }
    }
    Pthread_mutex_unlock(&ready_m_);
    if (timed_out_) {
        error_code_ = ETIMEDOUT;
        if (attr_.callback != nullptr) {
            attr_.callback(this);
        }
    }
}

void Future::notify_ready() {
    Pthread_mutex_lock(&ready_m_);
    if (!timed_out_) {
        ready_ = true;
    }
    Pthread_cond_signal(&ready_cond_);
    Pthread_mutex_unlock(&ready_m_);
    if (ready_ && attr_.callback != nullptr) {
        attr_.callback(this);
    }
}

Client::~Client() {
    if (udp_sa_ != nullptr) {
        free(udp_sa_);
    }
    invalidate_pending_futures();
}

void Client::invalidate_pending_futures() {
    list<Future*> futures;
    pending_fu_l_.lock();
    for (auto& it: pending_fu_) {
        futures.push_back(it.second);
    }
    pending_fu_.clear();
    pending_fu_l_.unlock();

    for (auto& fu: futures) {
        if (fu != nullptr) {
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
    }
    status_ = CLOSED;
    invalidate_pending_futures();
}

int Client::connect(const char* addr) {
    verify(status_ != CONNECTED);

    sock_ = tcp_connect(addr);
    if (sock_ == -1) {
        Log_error("rpc::Client: connect(%s): %s", addr, strerror(errno));
        goto err_out;
    }
    verify(set_nonblocking(sock_, true) == 0);

    udp_sock_ = udp_connect(addr, &udp_sa_, &udp_salen_);
    if (udp_sock_ == -1) {
        Log_error("rpc::Client: connect(%s): %s (UDP)", addr, strerror(errno));
        goto err_out;
    }

    Log_debug("rpc::Client: connected to %s", addr);
    status_ = CONNECTED;
    pollmgr_->add(this);

    return 0;

err_out:

    if (sock_ != -1) {
        ::close(sock_);
    }
    if (udp_sock_ != -1) {
        ::close(udp_sock_);
    }
    return ENOTCONN;
}

void Client::handle_error() {
    close();
}

void Client::handle_write() {
    if (status_ != CONNECTED) {
        return;
    }

    out_l_.lock();
    out_.write_to_fd(sock_);

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
        if (n_peek == sizeof(i32) && in_.content_size() >= packet_size + sizeof(i32)) {
            // consume the packet size
            verify(in_.read(&packet_size, sizeof(i32)) == sizeof(i32));

            v64 v_reply_xid;
            v32 v_error_code;

            in_ >> v_reply_xid >> v_error_code;

            pending_fu_l_.lock();
            unordered_map<i64, Future*>::iterator it = pending_fu_.find(v_reply_xid.get());
            if (it != pending_fu_.end()) {
                Future* fu = it->second;
                verify(fu->xid_ == v_reply_xid.get());
                pending_fu_.erase(it);
                pending_fu_l_.unlock();

                fu->error_code_ = v_error_code.get();
                fu->reply_.read_from_marshal(in_, packet_size - v_reply_xid.val_size() - v_error_code.val_size());

                fu->notify_ready();

                // since we removed it from pending_fu_
                fu->release();
            } else {
                // the future might timed out
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
        return nullptr;
    }

    Future* fu = new Future(xid_counter_.next(), attr);
    pending_fu_l_.lock();
    pending_fu_[fu->xid_] = fu;
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

        return nullptr;
    }

    bmark_ = out_.set_bookmark(sizeof(i32)); // will fill packet size later

    *this << v64(fu->xid_);
    *this << rpc_id;

    // one ref is already in pending_fu_
    return (Future *) fu->ref_copy();
}

void Client::end_request() {
    // set reply size in packet
    if (bmark_ != nullptr) {
        i32 request_size = out_.get_and_reset_write_cnt();
        out_.write_bookmark(bmark_, &request_size);
        delete bmark_;
        bmark_ = nullptr;
    }

    // always enable write events since the code above gauranteed there
    // will be some data to send
    pollmgr_->update_mode(this, Pollable::READ | Pollable::WRITE);

    out_l_.unlock();
}

// <size> <rpc_id> <arg1> <arg2> ... <argN>
void Client::begin_udp_request(i32 rpc_id) {
    udp_l_.lock();
    udp_bmark_ = udp_.base().set_bookmark(sizeof(i32)); // will fill packet size later
    udp_ << rpc_id;
}

int Client::end_udp_request() {
    i32 payload_size = udp_.base().get_and_reset_write_cnt();
    if (udp_bmark_ != nullptr) {
        udp_.base().write_bookmark(udp_bmark_, &payload_size);
        delete udp_bmark_;
        udp_bmark_ = nullptr;
    }

    int ret = 0;
    size_t size = 0;
    bool overflow = false;
    char* buf = udp_.get_buf(&size, &overflow);
    if (overflow) {
        ret = E2BIG;
    } else {
        sendto(udp_sock_, buf, size, 0, udp_sa_, udp_salen_);
    }
    udp_l_.unlock();
    return ret;
}


ClientPool::ClientPool(PollMgr* pollmgr /* =? */, int parallel_connections /* =? */)
        : parallel_connections_(parallel_connections) {

    verify(parallel_connections_ > 0);
    if (pollmgr == nullptr) {
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
    Client* cl = nullptr;
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
            insert_into_map(cache_, addr, parallel_clients);
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
