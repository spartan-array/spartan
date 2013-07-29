#ifndef COMMON_H_
#define COMMON_H_

#include "glog/logging.h"
#include "gflags/gflags.h"
#include <google/protobuf/message.h>

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
}

// operator<< overload to allow protocol buffers to be output from the logging methods.
namespace std {
inline ostream & operator<<(ostream &out, const google::protobuf::Message &q) {
  string s = q.ShortDebugString();
  out << s;
  return out;
}
}

#define COMPILE_ASSERT(x) extern int __dummy[(int)x]

#endif /* COMMON_H_ */
