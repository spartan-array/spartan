#ifndef COMMON_H_
#define COMMON_H_

#include <ostream>
#include <google/protobuf/message.h>
#include <vector>

namespace sparrow {

void Init(int argc, char** argv);

uint64_t get_memory_rss();
uint64_t get_memory_total();

void Sleep(double t);
void DumpProfile();

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

} // namespace sparrow

// operator<< overload to allow protocol buffers to be output from the logging methods.
namespace std {
inline ostream & operator<<(ostream &out, const google::protobuf::Message &q) {
  string s = q.ShortDebugString();
  out << s;
  return out;
}

template <class A, class B>
inline ostream & operator<<(ostream &out, const std::pair<A, B>& p) {
  out << "(" << p.first << "," << p.second << ")";
  return out;
}

}

#define COMPILE_ASSERT(x) extern int __dummy[(int)x]

#endif /* COMMON_H_ */
