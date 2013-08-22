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

void ThreadPool::run_thread(int id_in_pool) {
    bool should_stop = false;
    while (!should_stop) {
        list<function<void()>*>* jobs = q_[id_in_pool].pop_all();
        for (auto& f : *jobs) {
            if (f == nullptr) {
                should_stop = true;
            } else {
                (*f)();
                delete f;
            }
        }
        delete jobs;
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

void Log::log_v(int level, const char* file, int line, const char* fmt, va_list args) {
    static char indicator[] = { 'F', 'E', 'W', 'I', 'D' };
    assert(level <= Log::DEBUG);
    if (level <= Log::level) {
        Pthread_mutex_lock(&Log::m);
        const char* filebase = basename(file);
        fprintf(Log::fp, "%c ", indicator[level]);
        fprintf(Log::fp, "%s:%3d ", filebase, line);
        vfprintf(Log::fp, fmt, args);
        fprintf(Log::fp, "\n");
        fflush(Log::fp);
        Pthread_mutex_unlock(&Log::m);
    }
}

void Log::log(int level, const char* file, int line, const char* fmt, ...) {
    va_list args;
    va_start(args, fmt);
    log_v(level, file, line, fmt, args);
    va_end(args);
}

void Log::fatal(const char* file, int line, const char* fmt, ...) {
    va_list args;
    va_start(args, fmt);
    log_v(Log::FATAL, file, line, fmt, args);
    va_end(args);

    abort();
}

void Log::error(const char* file, int line, const char* fmt, ...) {
    va_list args;
    va_start(args, fmt);
    log_v(Log::ERROR, file, line, fmt, args);
    va_end(args);
}

void Log::warn(const char* file, int line, const char* fmt, ...) {
    va_list args;
    va_start(args, fmt);
    log_v(Log::WARN, file, line, fmt, args);
    va_end(args);
}

void Log::info(const char* file, int line, const char* fmt, ...) {
    va_list args;
    va_start(args, fmt);
    log_v(Log::INFO, file, line, fmt, args);
    va_end(args);
}

void Log::debug(const char* file, int line, const char* fmt, ...) {
    va_list args;
    va_start(args, fmt);
    log_v(Log::DEBUG, file, line, fmt, args);
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
