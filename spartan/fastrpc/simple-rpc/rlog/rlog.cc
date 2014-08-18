#include <string>
#include <stdio.h>
#include <stdlib.h>

#include "rlog.h"

using namespace std;
using namespace rpc;
using namespace rlog;

namespace rlog {

char* RLog::my_ident_s = nullptr;
RLogProxy* RLog::rp_s = nullptr;
Client* RLog::cl_s = nullptr;
char* RLog::buf_s = nullptr;
int RLog::buf_len_s = -1;
PollMgr* RLog::poll_s = nullptr;
rpc::Counter RLog::msg_counter_s;

// no static Mutex class, use pthread_mutex_t and PTHREAD_MUTEX_INITIALIZER instead
pthread_mutex_t RLog::mutex_s = PTHREAD_MUTEX_INITIALIZER;

void RLog::init(const char* my_ident /* =? */, const char* rlog_addr /* =? */) {
    Pthread_mutex_lock(&mutex_s);
    if (RLog::cl_s == nullptr) {
        if (my_ident == nullptr) {
            const int len = 128;
            char cstr[len];
            verify(gethostname(cstr, len) == 0);
            string src = cstr;
            sprintf(cstr, "(pid=%d)", getpid());
            src += cstr;
            RLog::my_ident_s = strdup(src.c_str());
        } else {
            RLog::my_ident_s = strdup(my_ident);
        }

        if (rlog_addr == nullptr) {
            rlog_addr = getenv("RLOG_SERVER");
        }
        if (rlog_addr == nullptr) {
            rlog_addr = getenv("RLOGSERVER");
        }
        if (poll_s == nullptr) {
            poll_s = new PollMgr;
        }
        cl_s = new Client(poll_s);
        if (rlog_addr == nullptr || cl_s->connect(rlog_addr) != 0) {
            Log_info("RLog working in local mode");
            do_finalize();
        } else {
            rp_s = new RLogProxy(cl_s);
            msg_counter_s.reset(0);
        }
    } else {
        Log_warn("called RLog::init() multiple times without calling RLog::finalize() first");
    }
    Pthread_mutex_unlock(&mutex_s);
}


// function called while holding lock on RLog
void RLog::do_finalize() {
    if (my_ident_s) {
        free(my_ident_s);
        my_ident_s = nullptr;
    }
    if (cl_s) {
        cl_s->close_and_release();
        cl_s = nullptr;
    }
    if (rp_s) {
        delete rp_s;
        rp_s = nullptr;
    }
    if (buf_s) {
        free(buf_s);
        buf_s = nullptr;
    }
    if (poll_s) {
        poll_s->release();
        poll_s = nullptr;
    }
}

void RLog::log_v(int level, const char* fmt, va_list args) {
    Pthread_mutex_lock(&mutex_s);
    if (buf_s == nullptr) {
        buf_len_s = 8192;
        buf_s = (char *) malloc(buf_len_s);
        memset(buf_s, 0, buf_len_s);
    }
    int cnt = vsnprintf(buf_s, buf_len_s - 1, fmt, args);
    if (cnt >= buf_len_s - 1) {
        buf_len_s = cnt + 16;
        buf_s = (char *) realloc(buf_s, buf_len_s);
        cnt = vsnprintf(buf_s, buf_len_s - 1, fmt, args);
        verify(cnt < buf_len_s - 1);
    }
    buf_s[cnt] = '\0';
    // TODO update remote logging
    LOG_INFO("level=%d, %s", level, buf_s);
    if (rp_s) {
        // always use async rpc
        string message = buf_s;
        Future* fu = rp_s->async_log(level, my_ident_s, msg_counter_s.next(), message);
        if (fu != nullptr) {
            fu->release();
        } else {
            Log_error("RLog connection failed, fall back to local mode");
            do_finalize();
        }
    }
    Pthread_mutex_unlock(&mutex_s);
}

void RLog::aggregate_qps(const std::string& metric_name, const rpc::i32 increment) {
    Pthread_mutex_lock(&mutex_s);
    if (rp_s) {
        // always use async rpc
        Future* fu = rp_s->async_aggregate_qps(metric_name, increment);
        if (fu != nullptr) {
            fu->release();
        } else {
            Log_error("RLog connection failed, cannot report qps");
            do_finalize();
        }
    }
    Pthread_mutex_unlock(&mutex_s);
}

} // namespace rlog
