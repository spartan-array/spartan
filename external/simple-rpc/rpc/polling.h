#pragma once

#include <map>
#include <set>

#include "utils.h"
#include "marshal.h"

namespace rpc {


struct poll_options {
    int n_threads;
    io_ratelimit rate;

    poll_options(): n_threads(1) {}
};


class Pollable: public RefCounted {
protected:

    virtual ~Pollable() {
    }

public:

    enum {
        READ = 0x1, WRITE = 0x2
    };

    virtual int fd() = 0;
    virtual int poll_mode() = 0;
    virtual void handle_read() = 0;
    virtual void handle_write(const io_ratelimit& rate) = 0;
    virtual void handle_error() = 0;
};

class PollMgr: public RefCounted {

    class PollThread;

    PollThread* poll_threads_;
    poll_options opts_;

protected:

    // RefCounted object uses protected dtor to prevent accidental deletion
    ~PollMgr();

public:

    PollMgr(const poll_options& opts = poll_options());

    void add(Pollable*);
    void remove(Pollable*);
    void update_mode(Pollable*, int new_mode);
};

}
