#ifndef TIMER_H_
#define TIMER_H_

namespace spartan {

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

class Timer {
public:
  Timer() {
    reset();
  }

  void reset() {
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

}

#define EVERY_N(interval, operation)\
{ static int COUNT = 0;\
  if (COUNT++ % interval == 0) {\
    operation;\
  }\
}

#define START_PERIODIC(interval)\
{ static Timer _t; \
  static int _count = 0; \
  ++_count; \
  if (_t.elapsed() > interval) {\
    _t.reset();

#define END_PERIODIC\
  }\
}

#define PERIODIC(interval, operation)\
    START_PERIODIC(interval)\
    operation;\
    END_PERIODIC

#endif /* TIMER_H_ */
