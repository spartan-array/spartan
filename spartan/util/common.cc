#include "spartan/util/common.h"

#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <signal.h>
#include <execinfo.h>
#include <fcntl.h>

#include <math.h>

#include <asm/msr.h>
#include <sys/time.h>
#include <sys/stat.h>
#include <sys/types.h>

#include "spartan-config.h"

#ifdef HAVE_LZO
#include <lzo/lzo1x.h>
#endif

namespace spartan {

double rand_double() {
  return double(random()) / RAND_MAX;
}


void Sleep(double t) {
  timespec req;
  req.tv_sec = (int) t;
  req.tv_nsec = (int64_t) (1e9 * (t - (int64_t) t));
  nanosleep(&req, NULL);
}

void SpinLock::lock() volatile {
  while (!__sync_bool_compare_and_swap(&d, 0, 1))
    ;
}

void SpinLock::unlock() volatile {
  d = 0;
}

void Init(int argc, char** argv) {
  if (!getenv("PYTHONPATH")) {
    setenv("PYTHONPATH", "", 0);
  }

  srandom(time(NULL));
}

void print_backtrace() {
  Log::error("Stack: ");
  void* stack[32];
  backtrace(stack, 32);
  backtrace_symbols_fd(stack, 32, STDERR_FILENO);
}

} // namespace spartan
