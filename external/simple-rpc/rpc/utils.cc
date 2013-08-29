#include <utility>

#include <fcntl.h>
#include <stdlib.h>
#include <sys/time.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <netdb.h>
#include <stdio.h>
#include <string.h>
#include <unistd.h>
#include "utils.h"

using namespace std;

namespace rpc {

struct start_thread_pool_args {
    ThreadPool* thrpool;
    int id_in_pool;
};

void* ThreadPool::start_thread_pool(void* args) {
    start_thread_pool_args* t_args = (start_thread_pool_args *) args;
    t_args->thrpool->run_thread(t_args->id_in_pool);
    delete t_args;
    pthread_exit(NULL);
    return NULL;
}

ThreadPool::ThreadPool(int n /* =... */)
        : n_(n) {
    th_ = new pthread_t[n_];
    q_ = new Queue<function<void()>*> [n_];

    for (int i = 0; i < n_; i++) {
        start_thread_pool_args* args = new start_thread_pool_args();
        args->thrpool = this;
        args->id_in_pool = i;
        Pthread_create(&th_[i], NULL, ThreadPool::start_thread_pool, args);
    }
}

ThreadPool::~ThreadPool() {
    for (int i = 0; i < n_; i++) {
        q_[i].push(nullptr);  // nullptr is used as a termination token
    }
    for (int i = 0; i < n_; i++) {
        Pthread_join(th_[i], nullptr);
    }
    delete[] th_;
    delete[] q_;
}

void ThreadPool::run_async(const std::function<void()>& f) {
    // Randomly select a thread for the job.
    // There could be better schedule policy.
    int queue_id = rand_engine_() % n_;
    q_[queue_id].push(new function<void()>(f));
}

// How long should a thread wait before stealing from
// other queues.
static const int kStealThreshold = 10 * 1000 * 1000;

void ThreadPool::run_thread(int id_in_pool) {
    bool should_stop = false;
    int64_t last_item_found = rdtsc();
    list<function<void()>*> jobs;
    struct timespec sleep_req;
    sleep_req.tv_nsec = 1;
    sleep_req.tv_sec = 0;

    while (!should_stop) {
        if (sleep_req.tv_nsec > 1) {
          nanosleep(&sleep_req, NULL);
        }

        int64_t now = rdtsc();
        if (now - last_item_found > kStealThreshold) {
          // start checking other queues
          for (auto i = 0; i < n_; ++i) {
            q_[i].pop_many(&jobs, 1);
            if (!jobs.empty()) {
              break;
            }
          }
        } else {
          q_[id_in_pool].pop_many(&jobs, 1);
        }

        if (jobs.empty()) {
            sleep_req.tv_nsec = std::min(1000000l, sleep_req.tv_nsec << 1);
            continue;
        }

        last_item_found = now;
        sleep_req.tv_nsec = 1;

        while (!jobs.empty()) {
            auto& f = jobs.front();
            if (f == nullptr) {
                should_stop = true;
            } else {
                (*f)();
                delete f;
            }
            jobs.pop_front();
        }
    }
}

int Log::level = Log::DEBUG;
FILE* Log::fp = stdout;
pthread_mutex_t Log::m = PTHREAD_MUTEX_INITIALIZER;

void Log::set_level(int level) {
    Pthread_mutex_lock(&Log::m);
    Log::level = level;
    Pthread_mutex_unlock(&Log::m);
}

void Log::set_file(FILE* fp) {
    verify(fp != NULL);
    Pthread_mutex_lock(&Log::m);
    Log::fp = fp;
    Pthread_mutex_unlock(&Log::m);
}

static const char* basename(const char* fpath) {
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
    verify(idx >= 0 && idx < len);
    return &fpath[idx];
}

void Log::log_v(int level, int line, const char* file, const char* fmt, va_list args) {
    static char indicator[] = { 'F', 'E', 'W', 'I', 'D' };
    assert(level <= Log::DEBUG);
    if (level <= Log::level) {
        const char* filebase = basename(file);

        const int tm_str_len = 80;
        char tm_str[tm_str_len];
        time_t now = time(NULL);
        struct tm tm_val;
        localtime_r(&now, &tm_val);
        strftime(tm_str, tm_str_len - 1, "%F %T", &tm_val);
        timeval tv;
        gettimeofday(&tv, NULL);

        Pthread_mutex_lock(&Log::m);
        fprintf(Log::fp, "%c ", indicator[level]);
        if (filebase != nullptr) {
            fprintf(Log::fp, "<%s:%d> ", filebase, line);
        }

        fprintf(Log::fp, "%s.%03d| ", tm_str, tv.tv_usec / 1000);
        vfprintf(Log::fp, fmt, args);
        fprintf(Log::fp, "\n");
        fflush(Log::fp);
        Pthread_mutex_unlock(&Log::m);
    }
}

void Log::log(int level, int line, const char* file, const char* fmt, ...) {
    va_list args;
    va_start(args, fmt);
    log_v(level, line, file, fmt, args);
    va_end(args);
}

void Log::fatal(int line, const char* file, const char* fmt, ...) {
    va_list args;
    va_start(args, fmt);
    log_v(Log::FATAL, line, file, fmt, args);
    va_end(args);
    abort();
}

void Log::error(int line, const char* file, const char* fmt, ...) {
    va_list args;
    va_start(args, fmt);
    log_v(Log::ERROR, line, file, fmt, args);
    va_end(args);
}

void Log::warn(int line, const char* file, const char* fmt, ...) {
    va_list args;
    va_start(args, fmt);
    log_v(Log::WARN, line, file, fmt, args);
    va_end(args);
}

void Log::info(int line, const char* file, const char* fmt, ...) {
    va_list args;
    va_start(args, fmt);
    log_v(Log::INFO, line, file, fmt, args);
    va_end(args);
}

void Log::debug(int line, const char* file, const char* fmt, ...) {
    va_list args;
    va_start(args, fmt);
    log_v(Log::DEBUG, line, file, fmt, args);
    va_end(args);
}


void Log::fatal(const char* fmt, ...) {
    va_list args;
    va_start(args, fmt);
    log_v(Log::FATAL, 0, nullptr, fmt, args);
    va_end(args);
    abort();
}

void Log::error(const char* fmt, ...) {
    va_list args;
    va_start(args, fmt);
    log_v(Log::ERROR, 0, nullptr, fmt, args);
    va_end(args);
}

void Log::warn(const char* fmt, ...) {
    va_list args;
    va_start(args, fmt);
    log_v(Log::WARN, 0, nullptr, fmt, args);
    va_end(args);
}

void Log::info(const char* fmt, ...) {
    va_list args;
    va_start(args, fmt);
    log_v(Log::INFO, 0, nullptr, fmt, args);
    va_end(args);
}

void Log::debug(const char* fmt, ...) {
    va_list args;
    va_start(args, fmt);
    log_v(Log::DEBUG, 0, nullptr, fmt, args);
    va_end(args);
}


int set_nonblocking(int fd, bool nonblocking) {
    int ret = fcntl(fd, F_GETFL, 0);
    if (ret != -1) {
        if (nonblocking) {
            ret = fcntl(fd, F_SETFL, ret | O_NONBLOCK);
        } else {
            ret = fcntl(fd, F_SETFL, ret & ~O_NONBLOCK);
        }
    }
    return ret;
}

Timer::Timer() {
  reset();
}

void Timer::reset() {
  start_.tv_sec = 0;
  start_.tv_usec = 0;
  end_.tv_sec = 0;
  end_.tv_usec = 0;
}

void Timer::start() {
  gettimeofday(&start_, NULL);
}

void Timer::end() {
  gettimeofday(&end_, NULL);
}

double Timer::elapsed() const {
  // assumes end_ >= start_

  double sec = 0;
  double usec = 0;
  if (end_.tv_usec < start_.tv_usec) {
    sec = end_.tv_sec - 1 - start_.tv_sec;
    usec = (end_.tv_usec + 1000000 - start_.tv_usec) / 1000000.0;
  } else {
    sec = end_.tv_sec - start_.tv_sec;
    usec = (end_.tv_usec - start_.tv_usec) / 1000000.0;
  }
  return sec+usec;
}

int find_open_port() {
  int fd = socket(AF_INET, SOCK_STREAM, 0);

  addrinfo *local_addr;

  addrinfo hints;
  bzero(&hints, sizeof(hints));
  hints.ai_family = AF_INET;
  hints.ai_socktype = SOCK_STREAM;
  hints.ai_flags = AI_PASSIVE;

  if (getaddrinfo("localhost", NULL, NULL, &local_addr) != 0) {
    Log_fatal("Failed to getaddrinfo");
  }

  int port = -1;

  for (int i = 1024; i < 65000; ++i) {
    ((sockaddr_in*)local_addr->ai_addr)->sin_port = i;
    if (::bind(fd, local_addr->ai_addr, local_addr->ai_addrlen) != 0) {
      continue;
    }

    sockaddr_in addr;
    socklen_t addrlen;
    if (getsockname(fd, (sockaddr*)&addr, &addrlen) != 0) {
      Log_fatal("Failed to get socket address");
    }

    port = i;
    break;
  }

  freeaddrinfo(local_addr);
  ::close(fd);

  if (port != -1) {
    Log_info("Found open port: %d", port);
    return port;
  }

  Log_fatal("Failed to find open port.");
  return -1;
}

std::string get_host_name() {
  char buffer[1024];
  if (gethostname(buffer, 1024) != 0) {
    Log_fatal("Failed to get hostname.");
  }

  return std::string(buffer);
}

}
