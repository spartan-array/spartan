#pragma once

#include <stdarg.h>

#include "rpc/server.h"
#include "rpc/client.h"
#include "rpc/utils.h"
#include "log_service.h"

namespace logservice {

class RLog {
public:
    static void init(const char* my_ident = NULL, const char* rlog_addr = NULL);

    static void finalize() {
        lock_s.lock();
        do_finalize();
        lock_s.unlock();
    }

    static void log(int level, const char* fmt, ...) {
        va_list args;
        va_start(args, fmt);
        log_v(level, fmt, args);
        va_end(args);
    }

    static void debug(const char* fmt, ...) {
        va_list args;
        va_start(args, fmt);
        log_v(rpc::Log::DEBUG, fmt, args);
        va_end(args);
    }

    static void info(const char* fmt, ...) {
        va_list args;
        va_start(args, fmt);
        log_v(rpc::Log::INFO, fmt, args);
        va_end(args);
    }

    static void warn(const char* fmt, ...) {
        va_list args;
        va_start(args, fmt);
        log_v(rpc::Log::WARN, fmt, args);
        va_end(args);
    }

    static void error(const char* fmt, ...) {
        va_list args;
        va_start(args, fmt);
        log_v(rpc::Log::ERROR, fmt, args);
        va_end(args);
    }

    static void fatal(const char* fmt, ...) {
        va_list args;
        va_start(args, fmt);
        log_v(rpc::Log::FATAL, fmt, args);
        va_end(args);
    }

    static void aggregate_qps(const std::string& metric_name, const rpc::i32 increment);

private:
    static void log_v(int level, const char* fmt, va_list args);

    static void do_finalize() {
        if (my_ident_s) {
            free(my_ident_s);
            my_ident_s = NULL;
        }
        if (cl_s) {
            cl_s->close_and_release();
            cl_s = NULL;
        }
        if (rp_s) {
            delete rp_s;
            rp_s = NULL;
        }
        if (buf_s) {
            free(buf_s);
            buf_s = NULL;
        }
        if (poll_s) {
            poll_s->release();
            poll_s = NULL;
        }
    }

    static char* my_ident_s;
    static RLogProxy* rp_s;
    static rpc::Client* cl_s;
    static char* buf_s;
    static int buf_len_s;
    static rpc::LongLock lock_s;
    static rpc::PollMgr* poll_s;
    static rpc::Counter msg_counter_s;
};

}
