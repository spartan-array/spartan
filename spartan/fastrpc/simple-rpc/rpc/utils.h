#pragma once

#include <list>
#include <map>
#include <unordered_map>
#include <functional>

#include <sys/types.h>
#include <sys/time.h>
#include <stdarg.h>
#include <stdio.h>
#include <assert.h>
#include <pthread.h>
#include <inttypes.h>
#include <sys/socket.h>

#include "base/all.h"

namespace rpc {

using base::i8;
using base::i16;
using base::i32;
using base::i64;
using base::v32;
using base::v64;
using base::NoCopy;
using base::Lockable;
using base::SpinLock;
using base::Mutex;
using base::ScopedLock;
using base::CondVar;
using base::Log;
using base::RefCounted;
using base::Queue;
using base::Counter;
using base::Timer;
using base::Rand;
using base::ThreadPool;
using base::insert_into_map;

int set_nonblocking(int fd, bool nonblocking);

int open_socket(const char* addr, const struct addrinfo* hints,
                std::function<bool(int, const struct sockaddr*, socklen_t)> filter = nullptr,
                struct sockaddr** p_addr = nullptr, socklen_t* p_len = nullptr);

int tcp_connect(const char* addr);
int udp_connect(const char* addr, struct sockaddr** p_addr = nullptr, socklen_t* p_len = nullptr);

int udp_bind(const char* addr);

}
