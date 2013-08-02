#ifndef TIMER_H_
#define TIMER_H_

namespace sparrow {
static uint64_t rdtsc() {
  uint32_t hi, lo;
  __asm__ __volatile__ ("rdtsc" : "=a"(lo), "=d"(hi));
  return (((uint64_t) hi) << 32) | ((uint64_t) lo);
}

inline double Now() {
  timespec tp;
  clock_gettime(CLOCK_MONOTONIC, &tp);
  return tp.tv_sec + 1e-9 * tp.tv_nsec;
}

double get_processor_frequency();

class Timer {
public:
  Timer() {
    Reset();
  }

  void Reset() {
    start_time_ = Now();
    start_cycle_ = rdtsc();
  }

  double elapsed() const {
    return Now() - start_time_;
  }

  uint64_t cycles_elapsed() const {
    return rdtsc() - start_cycle_;
  }

  // Rate at which an event occurs.
  double rate(int count) {
    return count / (Now() - start_time_);
  }

  double cycle_rate(int count) {
    return double(cycles_elapsed()) / count;
  }

private:
  double start_time_;
  uint64_t start_cycle_;
};

struct PeriodicTimer {
  int64_t interval_;
  int64_t last_;
  PeriodicTimer(double interval) :
      interval_((int64_t) (interval * get_processor_frequency())), last_(0) {
  }

  bool fire() {
    int64_t now = rdtsc();
    if (now - last_ > interval_) {
      last_ = now;
      return true;
    }
    return false;
  }
};

}

#define EVERY_N(interval, operation)\
{ static int COUNT = 0;\
  if (COUNT++ % interval == 0) {\
    operation;\
  }\
}

#define START_PERIODIC(interval)\
{ static int64_t last = 0;\
  static int64_t cycles = (int64_t)(interval * get_processor_frequency());\
  static int COUNT = 0; \
  ++COUNT; \
  int64_t now = rdtsc(); \
  if (now - last > cycles) {\
    last = now;\

#define END_PERIODIC\
    COUNT = 0;\
  }\
}

#define PERIODIC(interval, operation)\
    START_PERIODIC(interval)\
    operation;\
    END_PERIODIC

#endif /* TIMER_H_ */
