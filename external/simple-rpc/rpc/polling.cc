#ifdef __APPLE__
#define USE_KQUEUE
#endif

#ifdef USE_KQUEUE
#include <sys/event.h>
#else
#include <sys/epoll.h>
#endif

#include <unordered_map>
#include <unordered_set>

#include <unistd.h>
#include <string.h>
#include <errno.h>

#include "utils.h"
#include "polling.h"

using namespace std;

namespace rpc {

class PollMgr::PollThread {

    friend class PollMgr;

    PollMgr* poll_mgr_;

    // guard mode_ and poll_set_
    ShortLock l_;
    std::unordered_map<int, int> mode_;
    std::unordered_set<Pollable*> poll_set_;
    int poll_fd_;

    std::unordered_set<Pollable*> pending_remove_;
    ShortLock pending_remove_l_;

    pthread_t th_;
    bool stop_flag_;

    static void* start_poll_loop(void* arg) {
        PollThread* thiz = (PollThread *) arg;
        thiz->poll_loop();
        pthread_exit(NULL);
        return NULL;
    }

    void poll_loop();

    void start(PollMgr* poll_mgr) {
        poll_mgr_ = poll_mgr;
        Pthread_create(&th_, NULL, PollMgr::PollThread::start_poll_loop, this);
    }

public:

    PollThread(): poll_mgr_(nullptr), stop_flag_(false) {
#ifdef USE_KQUEUE
        poll_fd_ = kqueue();
#else
        poll_fd_ = epoll_create(10);    // arg ignored, any value > 0 will do
#endif
        verify(poll_fd_ != -1);
    }

    ~PollThread() {
        l_.lock();
        unordered_set<Pollable*> poll_set_copy = poll_set_;
        l_.unlock();

        for (unordered_set<Pollable*>::iterator it = poll_set_copy.begin(); it != poll_set_copy.end(); ++it) {
            remove(*it);
        }

        stop_flag_ = true;
        Pthread_join(th_, NULL);
    }

    void add(Pollable*);
    void remove(Pollable*);
    void update_mode(Pollable*, int new_mode);
};

PollMgr::PollMgr(const poll_options& opts /* =... */): opts_(opts) {
    if (opts_.rate.min_size > 0 && opts_.rate.interval <= 0.0) {
        Log::warn("rpc batching size set but wait time not set, will use 1ms");
        opts_.rate.interval = 0.001;
    }
    poll_threads_ = new PollThread[opts_.n_threads];
    for (int i = 0; i < opts_.n_threads; i++) {
        poll_threads_[i].start(this);
    }
    //Log::debug("rpc::PollMgr: start with %d thread", opts_.n_threads);
}

PollMgr::~PollMgr() {
    delete[] poll_threads_;
    //Log::debug("rpc::PollMgr: destroyed");
}

void PollMgr::PollThread::poll_loop() {
    while (!stop_flag_) {
        const int max_nev = 100;

#ifdef USE_KQUEUE

        struct kevent evlist[max_nev];
        struct timespec timeout;
        timeout.tv_sec = 0;
        timeout.tv_nsec = 50 * 1000 * 1000; // 0.05 sec

        int nev = kevent(poll_fd_, NULL, 0, evlist, max_nev, &timeout);

        for (int i = 0; i < nev; i++) {
            Pollable* poll = (Pollable *) evlist[i].udata;
            verify(poll != NULL);

            if (evlist[i].filter == EVFILT_READ) {
                poll->handle_read();
            }
            if (evlist[i].filter == EVFILT_WRITE) {
                poll->handle_write(poll_mgr_->opts_.rate);
            }

            // handle error after handle IO, so that we can at least process something
            if (evlist[i].flags & EV_EOF) {
                poll->handle_error();
            }
        }

#else

        struct epoll_event evlist[max_nev];
        int timeout = 50; // milli, 0.05 sec

        int nev = epoll_wait(poll_fd_, evlist, max_nev, timeout);

        if (stop_flag_) {
            break;
        }

        for (int i = 0; i < nev; i++) {
            Pollable* poll = (Pollable *) evlist[i].data.ptr;
            verify(poll != NULL);

            if (evlist[i].events & EPOLLIN) {
                poll->handle_read();
            }
            if (evlist[i].events & EPOLLOUT) {
                poll->handle_write(poll_mgr_->opts_.rate);
            }

            // handle error after handle IO, so that we can at least process something
            if (evlist[i].events & (EPOLLERR | EPOLLHUP | EPOLLRDHUP)) {
                poll->handle_error();
            }
        }

#endif

        // after each poll loop, remove uninterested pollables
        pending_remove_l_.lock();
        list<Pollable*> remove_poll(pending_remove_.begin(), pending_remove_.end());
        pending_remove_.clear();
        pending_remove_l_.unlock();

        for (list<Pollable*>::iterator it = remove_poll.begin(); it != remove_poll.end(); ++it) {
            Pollable* poll = *it;
            int fd = poll->fd();

            l_.lock();
            if (mode_.find(fd) == mode_.end()) {
                // NOTE: only remove the fd when it is not immediately added again
                // if the same fd is used again, mode_ will contains its info
#ifdef USE_KQUEUE

                struct kevent ev;

                bzero(&ev, sizeof(ev));
                ev.ident = fd;
                ev.flags = EV_DELETE;
                ev.filter = EVFILT_READ;
                kevent(poll_fd_, &ev, 1, NULL, 0, NULL);

                bzero(&ev, sizeof(ev));
                ev.ident = fd;
                ev.flags = EV_DELETE;
                ev.filter = EVFILT_WRITE;
                kevent(poll_fd_, &ev, 1, NULL, 0, NULL);

#else
                struct epoll_event ev;
                memset(&ev, 0, sizeof(ev));

                epoll_ctl(poll_fd_, EPOLL_CTL_DEL, fd, &ev);
#endif
            }
            l_.unlock();

            poll->release();
        }
    }

    // when stopping, release anything registered in pollmgr
    for (unordered_set<Pollable*>::iterator it = poll_set_.begin(); it != poll_set_.end(); ++it) {
        (*it)->release();
    }

    close(poll_fd_);
}

void PollMgr::PollThread::add(Pollable* poll) {
    poll->ref_copy();   // increase ref count

    int poll_mode = poll->poll_mode();
    int fd = poll->fd();

    l_.lock();

    // verify not exists
    verify(poll_set_.find(poll) == poll_set_.end());
    verify(mode_.find(fd) == mode_.end());

    // register pollable
    poll_set_.insert(poll);
    mode_[fd] = poll_mode;

    l_.unlock();

#ifdef USE_KQUEUE

    struct kevent ev;
    if (poll_mode & Pollable::READ) {
        bzero(&ev, sizeof(ev));
        ev.ident = fd;
        ev.flags = EV_ADD;
        ev.filter = EVFILT_READ;
        ev.udata = poll;
        verify(kevent(poll_fd_, &ev, 1, NULL, 0, NULL) == 0);
    }
    if (poll_mode & Pollable::WRITE) {
        bzero(&ev, sizeof(ev));
        ev.ident = fd;
        ev.flags = EV_ADD;
        ev.filter = EVFILT_WRITE;
        ev.udata = poll;
        verify(kevent(poll_fd_, &ev, 1, NULL, 0, NULL) == 0);
    }

#else

    struct epoll_event ev;
    memset(&ev, 0, sizeof(ev));

    ev.data.ptr = poll;
    ev.events = EPOLLIN | EPOLLRDHUP; // EPOLLERR and EPOLLHUP are included by default
    if (poll_mgr_->opts_.rate.min_size <= 0 && poll_mgr_->opts_.rate.interval <= 0.0) {
        // only enable EPOLLET when not batching
        ev.events |= EPOLLET;
    }

    if (poll_mode & Pollable::WRITE) {
        ev.events |= EPOLLOUT;
    }
    verify(epoll_ctl(poll_fd_, EPOLL_CTL_ADD, fd, &ev) == 0);

#endif
}

void PollMgr::PollThread::remove(Pollable* poll) {
    bool found = false;
    l_.lock();
    unordered_set<Pollable*>::iterator it = poll_set_.find(poll);
    if (it != poll_set_.end()) {
        found = true;
        assert(mode_.find(poll->fd()) != mode_.end());
        poll_set_.erase(poll);
        mode_.erase(poll->fd());
    } else {
        assert(mode_.find(poll->fd()) == mode_.end());
    }
    l_.unlock();

    if (found) {
        pending_remove_l_.lock();
        pending_remove_.insert(poll);
        pending_remove_l_.unlock();
    }
}

void PollMgr::PollThread::update_mode(Pollable* poll, int new_mode) {
    int fd = poll->fd();

    l_.lock();

    if (poll_set_.find(poll) == poll_set_.end()) {
        l_.unlock();
        return;
    }

    unordered_map<int, int>::iterator it = mode_.find(fd);
    verify(it != mode_.end());
    int old_mode = it->second;
    it->second = new_mode;

    if (new_mode != old_mode) {

#ifdef USE_KQUEUE

        struct kevent ev;
        if ((new_mode & Pollable::READ) && !(old_mode & Pollable::READ)) {
            // add READ
            bzero(&ev, sizeof(ev));
            ev.ident = fd;
            ev.udata = poll;
            ev.flags = EV_ADD;
            ev.filter = EVFILT_READ;
            verify(kevent(poll_fd_, &ev, 1, NULL, 0, NULL) == 0);
        }
        if (!(new_mode & Pollable::READ) && (old_mode & Pollable::READ)) {
            // del READ
            bzero(&ev, sizeof(ev));
            ev.ident = fd;
            ev.udata = poll;
            ev.flags = EV_DELETE;
            ev.filter = EVFILT_READ;
            verify(kevent(poll_fd_, &ev, 1, NULL, 0, NULL) == 0);
        }
        if ((new_mode & Pollable::WRITE) && !(old_mode & Pollable::WRITE)) {
            // add WRITE
            bzero(&ev, sizeof(ev));
            ev.ident = fd;
            ev.udata = poll;
            ev.flags = EV_ADD;
            ev.filter = EVFILT_WRITE;
            verify(kevent(poll_fd_, &ev, 1, NULL, 0, NULL) == 0);
        }
        if (!(new_mode & Pollable::WRITE) && (old_mode & Pollable::WRITE)) {
            // del WRITE
            bzero(&ev, sizeof(ev));
            ev.ident = fd;
            ev.udata = poll;
            ev.flags = EV_DELETE;
            ev.filter = EVFILT_WRITE;
            verify(kevent(poll_fd_, &ev, 1, NULL, 0, NULL) == 0);
        }

#else

        struct epoll_event ev;
        memset(&ev, 0, sizeof(ev));

        ev.data.ptr = poll;
        ev.events = EPOLLRDHUP;
        if (poll_mgr_->opts_.rate.min_size <= 0 && poll_mgr_->opts_.rate.interval <= 0.0) {
            // only enable EPOLLET when not batching
            ev.events |= EPOLLET;
        }
        if (new_mode & Pollable::READ) {
            ev.events |= EPOLLIN;
        }
        if (new_mode & Pollable::WRITE) {
            ev.events |= EPOLLOUT;
        }
        verify(epoll_ctl(poll_fd_, EPOLL_CTL_MOD, fd, &ev) == 0);

#endif

    }

    l_.unlock();
}

void PollMgr::add(Pollable* poll) {
    int fd = poll->fd();
    if (fd >= 0) {
        int tid = fd % opts_.n_threads;
        poll_threads_[tid].add(poll);
    }
}

void PollMgr::remove(Pollable* poll) {
    int fd = poll->fd();
    if (fd >= 0) {
        int tid = fd % opts_.n_threads;
        poll_threads_[tid].remove(poll);
    }
}

void PollMgr::update_mode(Pollable* poll, int new_mode) {
    int fd = poll->fd();
    if (fd >= 0) {
        int tid = fd % opts_.n_threads;
        poll_threads_[tid].update_mode(poll, new_mode);
    }
}

}
