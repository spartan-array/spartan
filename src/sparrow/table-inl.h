// DON'T INCLUDE DIRECTLY; included from table.h
// Various templatized versions of things that confuse SWIG, etc.

#ifndef TABLE_INL_H_
#define TABLE_INL_H_

#include <boost/utility/enable_if.hpp>

namespace sparrow {

template<class T>
class Modulo: public SharderT<T> {
  size_t shard_for_key(const T& k, int num_shards) const {
    return boost::hash_value(k) % num_shards;
  }
  DECLARE_REGISTRY_HELPER(Sharder, Modulo);
};
TMPL_DEFINE_REGISTRY_HELPER(Sharder, Modulo);


template<class V>
struct Min: public AccumulatorT<V> {
  void accumulate(V* current, const V& update) const {
    *current = std::min(*current, update);
  }

  DECLARE_REGISTRY_HELPER(Accumulator, Min);
};
TMPL_DEFINE_REGISTRY_HELPER(Accumulator, Min)

template<class V>
struct Max: public AccumulatorT<V> {
  void accumulate(V* current, const V& update) const {
    *current = std::max(*current, update);
  }
  DECLARE_REGISTRY_HELPER(Accumulator, Max);
};
TMPL_DEFINE_REGISTRY_HELPER(Accumulator, Max);

template<class V>
struct Sum: public AccumulatorT<V> {
  void accumulate(V* current, const V& update) const {
    *current += update;
  }
  DECLARE_REGISTRY_HELPER(Accumulator, Sum);
};
TMPL_DEFINE_REGISTRY_HELPER(Accumulator, Sum);

template<class V>
struct Replace: public AccumulatorT<V> {
  void accumulate(V* current, const V& update) const {
    *current = update;
  }
  DECLARE_REGISTRY_HELPER(Accumulator, Replace);
};
TMPL_DEFINE_REGISTRY_HELPER(Accumulator, Replace);

} // namespace sparrow


#endif /* TABLE_INL_H_ */
