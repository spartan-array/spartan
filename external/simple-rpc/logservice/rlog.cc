#include <string>
#include <stdio.h>
#include <stdlib.h>

#include "rlog.h"

using namespace std;
using namespace rpc;
using namespace logservice;

char* RLog::my_ident_s = NULL;
RLogProxy* RLog::rp_s = NULL;
Client* RLog::cl_s = NULL;
char* RLog::buf_s = NULL;
int RLog::buf_len_s = -1;
PollMgr* RLog::poll_s = NULL;
rpc::LongLock RLog::lock_s;
rpc::Counter RLog::msg_counter_s;


void RLog::init(const char* my_ident /* =? */, const char* rlog_addr /* =? */) {
    lock_s.lock();
    if (RLog::cl_s == NULL) {
        if (my_ident == NULL) {
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

        if (rlog_addr == NULL) {
            rlog_addr = getenv("RLOG_SERVER");
        }
        if (poll_s == NULL) {
            poll_s = new PollMgr;
        }
        cl_s = new Client(poll_s);
        if (rlog_addr == NULL || cl_s->connect(rlog_addr) != 0) {
            Log::info("RLog working in local mode");
            do_finalize();
        } else {
            rp_s = new RLogProxy(cl_s);
            msg_counter_s.reset(0);
        }
    } else {
        Log::warn("called RLog::init() multiple times without calling RLog::finalize() first");
    }
    lock_s.unlock();
}

void RLog::log_v(int level, const char* fmt, va_list args) {
    lock_s.lock();
    if (buf_s == NULL) {
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
    Log::log(level, "%s", buf_s);
    if (rp_s) {
        // always use async rpc
        string message = buf_s;
        Future* fu = rp_s->async_log(level, my_ident_s, msg_counter_s.next(), message);
        if (fu != NULL) {
            fu->release();
        } else {
            Log::error("RLog connection failed, fall back to local mode");
            do_finalize();
        }
    }
    lock_s.unlock();
}

void RLog::aggregate_qps(const std::string& metric_name, const rpc::i32 increment) {
    lock_s.lock();
    if (rp_s) {
        // always use async rpc
        Future* fu = rp_s->async_aggregate_qps(metric_name, increment);
        if (fu != NULL) {
            fu->release();
        } else {
//            Log::error("RLog connection failed, cannot report qps");
            do_finalize();
        }
//    } else {
//        Log::error("RLog not initialized, cannot report qps");
    }
    lock_s.unlock();
}