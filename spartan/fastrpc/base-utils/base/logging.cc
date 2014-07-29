#include <assert.h>
#include <stdarg.h>
#include <string.h>
#include <sys/time.h>
#include <pthread.h>

#include "misc.h"
#include "threading.h"
#include "logging.h"

namespace {

// Serialize printing logs to prevent mangled output in multithread applications.
//
// Using recursive mutex, because it might be possible that a Log operation
// is interrupted by a signal, and then inside the signal handler, Log is
// called again. Using recursive mutex prevents the thread from deadlocking
// itself. See issue 11 on github.com/santazhang/simple-rpc
#ifdef __APPLE__
static pthread_mutex_t log_mutex = PTHREAD_RECURSIVE_MUTEX_INITIALIZER;
#else
static pthread_mutex_t log_mutex = PTHREAD_RECURSIVE_MUTEX_INITIALIZER_NP;
#endif // __APPLE__

static const char* file_basename(const char* fpath) {
    if (fpath == nullptr) {
        return nullptr;
    }
    const char sep = '/';
    int len = strlen(fpath);
    int idx = len - 1;
    while (idx > 0) {
        if (fpath[idx - 1] == sep) {
            break;
        }
        idx--;
    }
    assert(idx >= 0 && idx < len);
    return &fpath[idx];
}

} // namespace


namespace base {

Log::~Log() {
    rep_->ref--;
    if (rep_->ref == 0) {
        if (rep_->buf != nullptr) {
            (*this)("%s", rep_->buf->str().c_str());
            delete rep_->buf;
        }
        delete rep_;
    }
}

void Log::vlog(const char* fmt, va_list va) {
    if (rep_->verbosity < LOG_LEVEL) return;

    const char* severity = rep_->lm->severity();
    const char* filebase = file_basename(rep_->file);
    char now_str[TIME_NOW_STR_SIZE];
    time_now_str(now_str);

    pthread_mutex_lock(&log_mutex);
    if (filebase != nullptr) {
        printf("%s %s %s:%d | %s[%d] ", severity, now_str, filebase, rep_->line, rep_->lm->hostname(), rep_->lm->pid());
    } else {
        printf("%s %s | %s[%d] ", severity, now_str, rep_->lm->hostname(), rep_->lm->pid());
    }
    vprintf(fmt, va);
    printf("\n");
    pthread_mutex_unlock(&log_mutex);

    if (strcmp(severity, "FATAL") == 0) {
        // print stack trace for fatal errors, and abort
        print_stack_trace();
        ::abort();
    }
}

void Log::debug(const char* fmt, ...) {
    va_list va;
    va_start(va, fmt);
    LOG_DEBUG.vlog(fmt, va);
    va_end(va);
}

void Log::info(const char* fmt, ...) {
    va_list va;
    va_start(va, fmt);
    LOG_INFO.vlog(fmt, va);
    va_end(va);
}

void Log::warn(const char* fmt, ...) {
    va_list va;
    va_start(va, fmt);
    LOG_WARN.vlog(fmt, va);
    va_end(va);
}

void Log::error(const char* fmt, ...) {
    va_list va;
    va_start(va, fmt);
    LOG_ERROR.vlog(fmt, va);
    va_end(va);
}

void Log::fatal(const char* fmt, ...) {
    va_list va;
    va_start(va, fmt);
    LOG_FATAL.vlog(fmt, va);
    va_end(va);
}

char* LogManager::s_hostname_ = NULL;

LogManager LogManager_DEBUG("DEBUG", do_not_create_your_own());
LogManager LogManager_INFO("INFO", do_not_create_your_own());
LogManager LogManager_WARN("WARN", do_not_create_your_own());
LogManager LogManager_ERROR("ERROR", do_not_create_your_own());
LogManager LogManager_FATAL("FATAL", do_not_create_your_own());

int LOG_LEVEL = 0;

} // namespace base

