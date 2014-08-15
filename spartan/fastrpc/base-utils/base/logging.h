#ifndef BASE_LOGGING_H_
#define BASE_LOGGING_H_

#include <stdarg.h>
#include <sstream>
#include <unistd.h>

#include "base/misc.h"
#include "base/debugging.h"

#define LOG_DEBUG base::LogManager_DEBUG.new_log(__FILE__, __LINE__, 0)
#define LOG_INFO base::LogManager_INFO.new_log(__FILE__, __LINE__, 1)
#define LOG_WARN base::LogManager_WARN.new_log(__FILE__, __LINE__, 2)
#define LOG_ERROR base::LogManager_ERROR.new_log(__FILE__, __LINE__, 3)
#define LOG_FATAL base::LogManager_FATAL.new_log(__FILE__, __LINE__, 4)
#define LOG_NULL \
    true ? void(0) : base::hack_for_conditional_logging() & LOG_INFO

#ifndef likely
#define likely(x)   __builtin_expect((x), 1)
#endif // likely

#ifndef unlikely
#define unlikely(x)   __builtin_expect((x), 0)
#endif // unlikely

/**
 * Use assert() when the test is only intended for debugging.
 * Use verify() when the test is crucial for both debug and release binary.
 */
#define verify(invariant) \
    !(unlikely(!(invariant))) ? void(0) : base::hack_for_conditional_logging() & LOG_FATAL \
        << "verify(" << (#invariant) << ") failed at " << __FILE__ << ':' << __LINE__ << " in function " << __FUNCTION__

// for compatibility
#define Log_debug LOG_DEBUG
#define Log_info LOG_INFO
#define Log_warn LOG_WARN
#define Log_error LOG_ERROR
#define Log_fatal LOG_FATAL

namespace base {

class LogManager;

class Log {
    void operator= (const Log&) = delete;
public:
    Log(LogManager* lm, const char* file, int line, int verbosity) {
        rep_ = new rep;
        rep_->lm = lm;
        rep_->file = file;
        rep_->line = line;
        rep_->verbosity = verbosity;
    }

    Log(const Log& l): rep_(l.rep_) {
        rep_->ref++;
    }

    ~Log();

    std::ostream* stream() {
        if (rep_->buf == nullptr) {
            rep_->buf = new std::ostringstream;
        }
        return rep_->buf;
    }

    Log& operator() (const char* fmt, ...) {
        va_list va;
        va_start(va, fmt);
        vlog(fmt, va);
        va_end(va);
        return *this;
    }

    // for compatibility
    static void debug(const char* fmt, ...);
    static void info(const char* fmt, ...);
    static void warn(const char* fmt, ...);
    static void error(const char* fmt, ...);
    static void fatal(const char* fmt, ...);

private:
    void vlog(const char* fmt, va_list va);

    struct rep {
        int ref;
        LogManager* lm;
        std::ostringstream* buf;
        const char* file;
        int line;
        int verbosity;

        rep(): ref(1), lm(nullptr), buf(nullptr), file(nullptr), line(0), verbosity(0) {
        }
    };

    rep* rep_;
};


template <class T>
Log operator<< (Log log, const T& t) {
    *log.stream() << t;
    return log;
}

struct do_not_create_your_own {
};

class LogManager {
    MAKE_NOCOPY(LogManager);
public:
    LogManager(const char* _severity, struct do_not_create_your_own): severity_(_severity) {
        if (s_hostname_ == NULL)
            sethost();
    }

    ~LogManager() {
        if (s_hostname_ != NULL) {
            delete s_hostname_;
            s_hostname_ = NULL;
        }
    }

    Log new_log(const char* file, int line, int verbosity) {
        return Log(this, file, line, verbosity);
    }

    const char* severity() const {
        return severity_;
    }

    static char* hostname() {
        return s_hostname_;
    }

    static pid_t pid() {
        return getpid();
    }

    static void sethost() {
        s_hostname_ = new char [100];
        gethostname(s_hostname_, 100);
    }

private:
    const char* severity_;

    static char* s_hostname_;
};

struct hack_for_conditional_logging {
    // from google-glog library
    void operator& (Log) {
    }
};

extern LogManager LogManager_DEBUG, LogManager_INFO, LogManager_WARN, LogManager_ERROR, LogManager_FATAL;

extern int LOG_LEVEL;

} // namespace base

#endif // BASE_LOGGING_H_
