// DON'T INCLUDE DIRECTLY; included from table.h
// Various templatized versions of things that confuse SWIG, etc.

#ifndef TABLE_INL_H_
#define TABLE_INL_H_

#include <boost/utility/enable_if.hpp>

namespace spartan {

template<class K, class V>
RemoteIterator<K, V>::RemoteIterator(Table *table, int shard,
    uint32_t fetch_num) :
    table_(table), shard_(shard), done_(false), fetch_num_(fetch_num) {
  request_.table = table->id();
  request_.shard = shard_;
  request_.count = fetch_num;
  request_.id = -1;
  index_ = 0;
  int target_worker = table->worker_for_shard(shard);

  pending_ = table->workers[target_worker]->async_get_iterator(request_);
  request_.id = response_.id;
}

template<class K, class V>
bool RemoteIterator<K, V>::done() {
  wait_for_fetch();
  return response_.done && index_ >= response_.results.size();
}

template <class K, class V>
void RemoteIterator<K, V>::wait_for_fetch() {
  if (pending_ != NULL) {
    pending_->wait();
    pending_->get_reply() >> response_;
    pending_ = NULL;
  }
}

template<class K, class V>
void RemoteIterator<K, V>::next() {
  wait_for_fetch();
  ++index_;
  int target_worker = table_->worker_for_shard(shard_);

  if (index_ >= response_.results.size()) {
    if (response_.done) {
      return;
    }

    pending_ = table_->workers[target_worker]->async_get_iterator(request_);
  }
}

template<class K, class V>
std::string RemoteIterator<K, V>::key_str() {
  wait_for_fetch();
  return response_.results[index_].key;
}

template<class K, class V>
std::string RemoteIterator<K, V>::value_str() {
  wait_for_fetch();
  return response_.results[index_].value;
}

template<class T>
class Modulo: public SharderT<T> {
  size_t shard_for_key(const T& k, int num_shards) const {
    return boost::hash_value(k) % num_shards;
  }
  DECLARE_REGISTRY_HELPER(Sharder, Modulo);
};
TMPL_DEFINE_REGISTRY_HELPER(Sharder, Modulo);


}
 // namespace spartan

#endif /* TABLE_INL_H_ */
