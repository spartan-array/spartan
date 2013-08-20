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
  client->connect(addr.c_str());
  return new T(client);
}

#define CHECK(expr) if (!(expr)) { Log::fatal("Check failed: %s.", #expr); }
#define CHECK_EQ(a, b) CHECK((a == b))
#define CHECK_NE(a, b) CHECK((a != b))

void Sleep(double t);

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
