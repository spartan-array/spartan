#ifndef UTIL_STATS_H_
#define UTIL_STATS_H_

#include <string>
#include <map>
#include <vector>
#include "stringpiece.h"

namespace spartan {

// Simple wrapper around a string->double map.
struct Stats {
  double& operator[](const std::string& key);
  std::string ToString(std::string prefix);
  void Merge(Stats &other);
private:
  std::map<std::string, double> p_;
};

inline double& Stats::operator[](const std::string& key) {
  return p_[key];
}

inline std::string Stats::ToString(std::string prefix) {
  std::string out;
  for (std::map<std::string, double>::iterator i = p_.begin(); i != p_.end();
      ++i) {
    out += StringPrintf("%s -- %s : %.2f\n", prefix.c_str(), i->first.c_str(),
                        i->second);
  }
  return out;
}

inline void Stats::Merge(Stats &other) {
  for (std::map<std::string, double>::iterator i = other.p_.begin();
      i != other.p_.end(); ++i) {
    p_[i->first] += i->second;
  }
}


// Log-bucketed histogram.
class Histogram {
public:
  Histogram() : count(0) {}

  void add(double val);
  std::string summary();

  int bucketForVal(double v);
  double valForBucket(int b);

  int getCount() { return count; }
private:

  int count;
  std::vector<int> buckets;
  static const double kMinVal;
  static const double kLogBase;
};

}

#endif /* STATS_H_ */
