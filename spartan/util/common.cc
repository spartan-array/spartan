#include "spartan/util/common.h"
#include "spartan/util/stats.h"

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

const double Histogram::kMinVal = 1e-9;
const double Histogram::kLogBase = 1.1;

double rand_double() {
  return double(random()) / RAND_MAX;
}

int Histogram::bucketForVal(double v) {
  if (v < kMinVal) {
    return 0;
  }

  v /= kMinVal;
  v += kLogBase;

  return 1 + static_cast<int>(log(v) / log(kLogBase));
}

double Histogram::valForBucket(int b) {
  if (b == 0) {
    return 0;
  }
  return exp(log(kLogBase) * (b - 1)) * kMinVal;
}

void Histogram::add(double val) {
  int b = bucketForVal(val);
//  LOG_EVERY_N(INFO, 1000) << "Adding... " << val << " : " << b;
  if (buckets.size() <= b) {
    buckets.resize(b + 1);
  }
  ++buckets[b];
  ++count;
}

void DumpProfile() {
#ifdef HAVE_GOOGLE_PROFILER_H
  ProfilerFlush();
#endif
}

string Histogram::summary() {
  string out;
  int total = 0;
  for (int i = 0; i < buckets.size(); ++i) {
    total += buckets[i];
  }
  string hashes = string(100, '#');

  for (int i = 0; i < buckets.size(); ++i) {
    if (buckets[i] == 0) {
      continue;
    }
    out += StringPrintf("%-20.3g %6d %.*s\n", valForBucket(i), buckets[i],
        buckets[i] * 80 / total, hashes.c_str());
  }
  return out;
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
}
