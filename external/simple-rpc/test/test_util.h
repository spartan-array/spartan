#ifndef TEST_UTIL_HEADER
#define TEST_UTIL_HEADER

namespace test {

// A simple counter class used often in unit test (and not supposed to
// be used outside *_test.cpp files.

class Counter {
public:
  Counter()          { reset(); }
  ~Counter()         { }

  int  count() const { return count_; }
  void set(int i)    { count_ = i; }
  void reset()       { count_ = 0; }
  void inc()         { ++count_; }
  void incBy(int i)  { count_ += i; }

  bool between(int i, int j) {
    if (i > count_) return false;
    if (j < count_) return false;
    return true;
  }

private:
  int  count_;
};

}  // namespace base

#endif // TEST_UTIL_HEADER
