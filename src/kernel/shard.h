#ifndef LOCALTABLE_H_
#define LOCALTABLE_H_

#include "kernel/table.h"
#include "util/sparrow-val.h"
#include "util/marshal.h"
#include "util/file.h"
#include "util/rpc.h"
#include "util/timer.h"

#include <boost/unordered_map.hpp>

namespace sparrow {

template<class Key, class Value>
class HashShard: public Shard, public Checkpointable {
private:
  typedef boost::unordered_map<Key, Value> Map;
  Map map_;
  int id_;

public:

  HashShard(int id) :
      id_(id) {

  }

  ~HashShard() {
  }

  struct Iterator: public TableIterator {
    typename Map::iterator begin_;
    typename Map::iterator end_;
    typename Map::iterator cur_;

    Iterator(HashShard& parent) :
        begin_(parent.map_.begin()), end_(parent.map_.end()), cur_(
            parent.map_.begin()) {
    }

    void Next() {
      ++cur_;
    }

    bool done() {
      return cur_ == end_;
    }

    const TableValue& key() {
      return cur_->first;
    }

    const TableValue& value() {
      return cur_->second;
    }
  };

  void start_checkpoint(const string& f, bool delta);
  void finish_checkpoint();
  void restore(const string& f);
  void write_delta(const TableData& put);

  SparrowVal get(const Key& k);
  bool contains(const Key& k);
  void put(const Key& k, const Value& v);
  void update(const Key& k, const Value& v, const Accumulator& accum);

  void remove(const Key& k) {
    typename Map::iterator i = map_.find(k);
    if (i != map_.end()) {
      map_.erase(i);
    }
  }

  void resize(int64_t size);

  bool empty() {
    return size() == 0;
  }
  int64_t size() {
    return map_.size();
  }

  void clear() {
    map_.clear();
  }

  TableIterator *get_iterator() {
    return new Iterator(*this);
  }

private:
};

template<class Key, class Value>
bool HashShard<Key, Value>::contains(const Key& k) {
  return map_.find(k) != map_.end();
}

template<class Key, class Value>
SparrowVal HashShard<Key, Value>::get(const Key& k) {
  return map_[k];
}

template<class Key, class Value>
void HashShard<Key, Value>::update(const Key& k, const Value& v,
    const Accumulator& accum) {
  if (map_.find(k) == map_.end()) {
    map_[k] = v;
    return;
  }

  ((const AccumulatorT<Value>&) accum).accumulate(map_[k], v);
}

template<class Key, class Value>
void HashShard<Key, Value>::put(const Key& k, const Value& v) {
  map_[k] = v;
}

template<class Key, class Value>
void HashShard<Key, Value>::start_checkpoint(const string& f, bool delta) {
  VLOG(1) << "Start checkpoint " << f;
  Timer t;
  if (!delta) {
    FileWriter w(f);

    TableIterator* it = get_iterator();
    for (; !it->done(); it->next()) {
      it->key().write(&w);
      it->value().write(&w);
    }
  }
}

template<class Key, class Value>
void HashShard<Key, Value>::finish_checkpoint() {
}

template<class Key, class Value>
static bool read_entry(Reader* r, Key *k, Value *v) {
  if (!TableValue::read(k, r)) {
    return false;
  }

  if (!TableValue::read(v, r)) {
    return false;
  }

  return true;
}

template<class Key, class Value>
void HashShard<Key, Value>::restore(const string& f) {
  Key k;
  Value v;

  if (!File::Exists(f)) {
    //this is no longer a return-able condition because there might
    //be epochs that are just deltas for continuous checkpointing
    VLOG(2) << "Skipping full restore of non-existent shard " << f;
  } else {

    VLOG(2) << "Restoring full snapshot " << f;
    Timer t;

    FileReader r(f);
    while (read_entry(&r, &k, &v)) {
      put(k, v);
    }
  }
}

template<class Key, class Value>
void HashShard<Key, Value>::write_delta(const TableData& put) {
  LOG(FATAL)<< "Not implemented.";
}

}

#endif /* LOCALTABLE_H_ */
