#pragma once

#include <stdarg.h>

#include "rpc/server.h"
#include "rpc/client.h"
#include "rpc/utils.h"
#include "log_service.h"

namespace rlog {

class RLog {
public:
    static void init(const char* my_ident = nullptr, const char* rlog_addr = nullptr);

    static void finalize() {
        Pthread_mutex_lock(&mutex_s);
        do_finalize();
        Pthread_mutex_unlock(&mutex_s);
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
        const int level_debug = 4;
        log_v(level_debug, fmt, args);
        va_end(args);
    }

    static void info(const char* fmt, ...) {
        va_list args;
        va_start(args, fmt);
        const int level_info = 3;
        log_v(level_info, fmt, args);
        va_end(args);
    }

    static void warn(const char* fmt, ...) {
        va_list args;
        va_start(args, fmt);
        const int level_warn = 2;
        log_v(level_warn, fmt, args);
        va_end(args);
    }

    static void error(const char* fmt, ...) {
        va_list args;
        va_start(args, fmt);
        const int level_error = 1;
        log_v(level_error, fmt, args);
        va_end(args);
    }

    static void fatal(const char* fmt, ...) {
        va_list args;
        va_start(args, fmt);
        const int level_fatal = 0;
        log_v(level_fatal, fmt, args);
        va_end(args);
    }

    static void aggregate_qps(const std::string& metric_name, const rpc::i32 increment);

private:
    // prevent creating objects of RLog class, it's only a utility class with static functions
    RLog();

    static void log_v(int level, const char* fmt, va_list args);

    static void do_finalize();

    static char* my_ident_s;
    static RLogProxy* rp_s;
    static rpc::Client* cl_s;
    static char* buf_s;
    static int buf_len_s;
    static rpc::PollMgr* poll_s;
    static rpc::Counter msg_counter_s;

    // no static Mutex class, use pthread_mutex_t and PTHREAD_MUTEX_INITIALIZER instead
    static pthread_mutex_t mutex_s;
};

} // namespace rlog
