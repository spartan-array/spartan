#ifndef COMMON_H_
#define COMMON_H_

#include <ostream>
#include <vector>
#include "rpc/client.h"
#include "rpc/utils.h"

namespace spartan {

using rpc::Log;

template<class T>
T* connect(rpc::PollMgr* poller, std::string addr) {
  auto client = new rpc::Client(poller);
  if (client->connect(addr.c_str()) != 0) {
    Log_fatal("Failed to connect to host: %s", addr.c_str());
  }
  return new T(client);
}

void print_backtrace();

#define CHECK(expr) if (!(expr)) { Log_fatal("Check failed: %s.", #expr); }
#define CHECK_EQ(a, b) CHECK((a == b))
#define CHECK_NE(a, b) CHECK((a != b))
#define CHECK_LT(a, b) CHECK((a < b))
#define CHECK_GT(a, b) CHECK((a > b))

void Sleep(double t);

class SleepBackoff {
public:
  SleepBackoff(double max_time) : max_time_(max_time) {
  }

  void reset() {
    sleep_time_ = 1e-9;
  }

  void sleep() {
    Sleep(sleep_time_);
    sleep_time_ = std::min(sleep_time_ * 1.1, max_time_);
  }

private:
 double sleep_time_;
 double max_time_;
};

class SpinLock {
public:
  SpinLock() :
      d(0) {
  }
  void lock() volatile;
  void unlock() volatile;
private:
  volatile int d;
};

double rand_double();

inline std::vector<int> range(int from, int to) {
  std::vector<int> out;
  for (int i = from; i < to; ++i) {
    out.push_back(i);
  }
  return out;
}

inline std::vector<int> range(int to) {
  return range(0, to);
}

} // namespace spartan

namespace std {
template<class A, class B>
inline ostream & operator<<(ostream &out, const std::pair<A, B>& p) {
  out << "(" << p.first << "," << p.second << ")";
  return out;
}
}

#define COMPILE_ASSERT(x) extern int __dummy[(int)x]

#endif /* COMMON_H_ */
