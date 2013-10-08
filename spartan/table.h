#ifndef SPARTAN_TABLE_H
#define SPARTAN_TABLE_H

#include <map>
#include <queue>

#include <boost/functional/hash.hpp>
#include <boost/smart_ptr.hpp>
#include <boost/tuple/tuple.hpp>
#include <boost/unordered_map.hpp>

#include "spartan/util/common.h"
#include "spartan/util/marshal.h"
#include "spartan/util/registry.h"
#include "spartan/util/timer.h"
#include "spartan/spartan_service.h"

namespace spartan {

using std::make_pair;
using boost::make_tuple;

// How many entries to prefetch for remote iterators.
// TODO(power) -- this should be changed to always fetch
// X MB.
static const int kDefaultIteratorFetch = 2048;

#define GLOBAL_TABLE_USE_SCOPEDLOCK 1

#if GLOBAL_TABLE_USE_SCOPEDLOCK == 0
#define GRAB_LOCK do { } while(0)
#else
#define GRAB_LOCK rpc::ScopedLock sl(mutex())
#endif

// An instance of Marshal must be available for key and value types.
namespace val {

template<class T>
bool read(T* v, StringPiece src) {
  StringReader r(src);
  return Marshal<T>::read_value(&r, v);
}

template<class T>
void write(const T& v, string* out) {
  StringWriter w(out);
  Marshal<T>::write_value(&w, v);
}

template<class T>
std::string to_str(const T& v) {
  std::string out;
  write(v, &out);
  return out;
}

template<class T>
T from_str(const std::string& vstr) {
  T out;
  read(&out, vstr);
  return out;
}

} // namespace val

class Table;

class Sharder: public Initable {
public:
};

template<class K, class V>
class AccumulatorT;

class Accumulator: public Initable {
public:
  template<class K, class V>
  AccumulatorT<K, V>* cast() {
    return (AccumulatorT<K, V>*) this;
  }
};

class Selector: public Initable {
public:
};

template<class K, class V>
class AccumulatorT: public Accumulator {
public:
  virtual ~AccumulatorT() {
  }
  virtual void accumulate(const K& key, V* v, const V& update) const = 0;
};

template<class T>
class SharderT: public Sharder {
public:
  virtual ~SharderT() {
  }
  virtual size_t shard_for_key(const T& k, int num_shards) const = 0;
};

template<class K, class V>
class SelectorT: public Selector {
public:
  virtual ~SelectorT() {
  }

  virtual V select(const K& k, const V& v) = 0;
};

struct TableContext {
private:
public:
  virtual ~TableContext() {
  }

  virtual int id() const = 0;
  virtual Table* get_table(int id) const = 0;

  // Return the current table-context.  Thread-local.
  static TableContext* get_context();

  // Assign the given context for this thread.
  static void set_context(TableContext* ctx);
};

class TableIterator {
private:
public:
  virtual ~TableIterator() {
  }
  virtual std::string key_str() = 0;
  virtual std::string value_str() = 0;
  virtual int shard() = 0;
  virtual bool done() = 0;
  virtual void next() = 0;
};

template<class K, class V>
class TypedIterator: public TableIterator {
public:
  virtual ~TypedIterator() {
  }
  virtual K key() = 0;
  virtual V value() = 0;
};

class Checkpointable {
public:
  virtual ~Checkpointable() {
  }
  virtual void start_checkpoint(const std::string& f, bool delta) = 0;
  virtual void finish_checkpoint() = 0;
  virtual void restore(const std::string& f) = 0;
};

class Shard {
public:
  virtual ~Shard() {
  }
  virtual size_t size() const = 0;
};

template<class K, class V>
class TableT;

class Master;

class Table {
protected:
  std::vector<PartitionInfo> shard_info_;
  std::vector<Shard*> shards_;
  int id_;
  int pending_writes_;
  TableContext *ctx_;

public:
  std::vector<WorkerProxy*> workers;
  Sharder *sharder;
  Accumulator *combiner;
  Accumulator *reducer;
  Selector *selector;

  virtual ~Table() {
  }

  int pending_writes() {
    return pending_writes_;
  }

  void set_ctx(TableContext* h) {
    ctx_ = h;
  }

  TableContext* ctx() const {
    return ctx_;
  }

  // Convenience method for getting a handle to the master (when
  // running on the master host!)
  Master* master() const {
    return reinterpret_cast<Master*>(ctx_);
  }

  bool tainted(int shard) {
    return shard_info_[shard].tainted;
  }

  int worker_for_shard(int shard) const {
    return shard_info_[shard].owner;
  }

  bool is_local_shard(int shard) const {
    return worker_for_shard(shard) == ctx()->id();
  }

  int num_shards() const {
    return shard_info_.size();
  }

  int id() const {
    return id_;
  }

  Shard* shard(int id) {
    return shards_[id];
  }

  PartitionInfo* shard_info(int id) {
    return &shard_info_[id];
  }

  int64_t shard_size(int shard) {
    if (is_local_shard(shard)) {
      return shards_[shard]->size();
    } else {
      return shard_info_[shard].entries;
    }
  }

  template<class K, class V>
  TableT<K, V>* cast() {
    return (TableT<K, V>*) (this);
  }

  virtual void init(int id, int num_shards) = 0;

  virtual TableIterator* get_iterator(int shard_id) = 0;
  virtual int flush() = 0;

  // Untyped (string, string) operations.
  virtual std::string get_str(int shard, const std::string&) = 0;
  virtual bool contains_str(int shard, const std::string&) = 0;
  virtual void update_str(int shard, const std::string&,
      const std::string&) = 0;
};

template<class K, class V>
class RemoteIterator: public TypedIterator<K, V> {
public:
  RemoteIterator(Table *table, int shard, uint32_t fetch_num =
      kDefaultIteratorFetch);

  bool done();
  void next();

  std::string key_str();
  std::string value_str();

  K key() {
    return val::from_str<K>(key_str());
  }

  V value() {
    return val::from_str<V>(value_str());
  }

  int shard() {
    return shard_;
  }

private:
  void wait_for_fetch();

  Table* table_;
  rpc::Future* pending_;
  IteratorReq request_;
  IteratorResp response_;

  int shard_;
  int index_;
  bool done_;

  size_t fetch_num_;
};

template<class K, class V>
class ShardT: public Shard {
private:
  typedef boost::unordered_multimap<K, V> Map;
  Map data_;
public:

  virtual ~ShardT() {

  }
  typedef typename Map::iterator iterator;
  iterator begin() {
    return data_.begin();
  }

  iterator end() {
    return data_.end();
  }

  iterator find(const K& k) {
    return data_.find(k);
  }

  void insert(const K& k, const V& v) {
    CHECK_NE(k.get(), NULL);

    data_.insert(make_pair(k, v));
  }

  bool empty() const {
    return data_.empty();
  }

  void clear() {
    data_.clear();
  }

  const V& get(const K& k) {
    return data_.find(k)->second;
  }

  size_t size() const {
    return data_.size();
  }
};

template<class K, class V>
class LocalIterator: public TypedIterator<K, V> {
private:
  typename ShardT<K, V>::iterator begin_;
  typename ShardT<K, V>::iterator cur_;
  typename ShardT<K, V>::iterator end_;

  int shard_;
public:
  LocalIterator(int shard, ShardT<K, V>& m) :
      begin_(m.begin()), cur_(m.begin()), end_(m.end()), shard_(shard) {

  }

  int shard() {
    return shard_;
  }

  void next() {
    ++cur_;
  }

  bool done() {
    return cur_ == end_;
  }

  std::string key_str() {
    return val::to_str(cur_->first);
  }

  std::string value_str() {
    return val::to_str(cur_->second);
  }

  K key() {
    return cur_->first;
  }

  V value() {
    return cur_->second;
  }
};

template<class K, class V>
class MergedIterator: public TypedIterator<K, V> {
private:
  typedef std::vector<TypedIterator<K, V>*> IterList;
  IterList iters_;

  TypedIterator<K, V>* cur() {
    return iters_.back();
  }

public:
  MergedIterator(IterList iters) :
      iters_(iters) {
    while (!iters_.empty() && cur()->done()) {
      iters_.pop_back();
    }
  }

  void next() {
    cur()->next();
    while (!iters_.empty() && cur()->done()) {
      iters_.pop_back();
    }
  }

  int shard() {
    return cur()->shard();
  }

  bool done() {
    return iters_.empty();
  }

  std::string key_str() {
    return cur()->key_str();
  }

  std::string value_str() {
    return cur()->value_str();
  }

  K key() {
    return cur()->key();
  }

  V value() {
    return cur()->value();
  }
};

template<class K, class V>
class TableT: public Table {
public:
  typedef TypedIterator<K, V> Iterator;

  struct CacheEntry {
    double last_read_time;
    V val;
  };

private:
  boost::unordered_map<K, CacheEntry> cache_;
  rpc::Mutex m_;
  static TypeRegistry<Table>::Helper<TableT<K, V> > register_me_;

public:
  int type_id() {
    return register_me_.id();
  }

  void init(int id, int num_shards) {
    id_ = id;
    sharder = NULL;
    pending_writes_ = 0;
    ctx_ = NULL;

    shards_.resize(num_shards);
    shard_info_.resize(num_shards);
    for (int i = 0; i < num_shards; ++i) {
      shards_[i] = new ShardT<K, V>;
      shard_info_[i].owner = -1;
      shard_info_[i].shard = i;
      shard_info_[i].table = id;
    }

    combiner = reducer = NULL;
  }

  virtual ~TableT() {
    //Log_info("Deleting table %d", id());
    for (auto p : shards_) {
      delete p;
    }
  }

  bool is_local_key(const K& key) {
    return is_local_shard(shard_for_key(key));
  }

  int shard_for_key(const K& k) {
    CHECK_NE(sharder, NULL);
    return ((SharderT<K>*) sharder)->shard_for_key(k, this->num_shards());
  }

  ShardT<K, V>& typed_shard(int id) {
    return *((ShardT<K, V>*) this->shards_[id]);
  }

  void update(int shard, const K& k, const V& v) {
    if (shard == -1) {
      shard = this->shard_for_key(k);
    }

    CHECK_LT(shard, num_shards());

    ShardT<K, V>& s = typed_shard(shard);
    typename ShardT<K, V>::iterator i = s.find(k);

    GRAB_LOCK;
    if (is_local_shard(shard)) {
      CHECK_NE(k.get(), NULL);
      if (i == s.end() || reducer == NULL) {
        s.insert(k, v);
      } else {
        reducer->cast<K, V>()->accumulate(k, &i->second, v);
        CHECK_NE(i->second.get(), NULL);
      }
      return;
    }

    if (combiner != NULL) {
      if (i == s.end()) {
        s.insert(k, v);
      } else {
        combiner->cast<K, V>()->accumulate(k, &i->second, v);
      }

      return;
    }

    TableData put;
    put.table = this->id();
    put.shard = shard;
    put.kv_data.push_back( { val::to_str(k), val::to_str(v) });
    workers[worker_for_shard(shard)]->async_put(put);
  }

  bool _get(int shard, const K& k, V* v) {
    if (shard == -1) {
      shard = this->shard_for_key(k);
    }

    while (tainted(shard)) {
      sched_yield();
    }

    if (is_local_shard(shard)) {
      GRAB_LOCK;
      ShardT<K, V>& s = (ShardT<K, V>&) (*shards_[shard]);
      typename ShardT<K, V>::iterator i = s.find(k);
      if (i == s.end()) {
        return false;
      }

      if (v != NULL) {
        if (selector != NULL) {
          *v = ((SelectorT<K, V>*) selector)->select(k, i->second);
        } else {
          *v = i->second;
        }
      }

      return true;
    }

    return get_remote(shard, k, v);
  }

  V get(int shard, const K& k) {
    V out;
    _get(shard, k, &out);
    return out;
  }

  bool contains(int shard, const K& k) {
    return _get(shard, k, NULL);
  }

  void remove(const K& k) {
    Log_fatal("Not implemented!");
  }

  // Untyped operations:
  bool contains_str(int shard, const std::string& k) {
    return contains(shard, val::from_str<K>(k));
  }

  std::string get_str(int shard, const std::string& k) {
    return val::to_str(get(shard, val::from_str<K>(k)));
  }

  void update_str(int shard, const std::string& k, const std::string& v) {
    update(shard, val::from_str<K>(k), val::from_str<V>(v));
  }

  Shard* create_local(int shard_id) {
    return new ShardT<K, V>();
  }

  TypedIterator<K, V>* get_iterator() {
    std::vector<TypedIterator<K, V>*> iters;
    for (int i = 0; i < num_shards(); ++i) {
      iters.push_back(get_iterator(i));
    }
    return new MergedIterator<K, V>(iters);
  }

  TypedIterator<K, V>* get_iterator(int shard) {
    if (this->is_local_shard(shard)) {
      auto s = shards_[shard];
      return new LocalIterator<K, V>(shard, *((ShardT<K, V>*)s));
    } else {
      return new RemoteIterator<K, V>(this, shard);
    }
  }

  void update_partitions(const PartitionInfo& info) {
    shard_info_[info.shard] = info;
  }

  bool is_local_shard(int shard) {
    return worker_for_shard(shard) == ctx()->id();
  }

  bool get_remote(int shard, const K& k, V* v) {
    GetRequest req;
    TableData resp;

    req.key = val::to_str(k);
    req.table = id();
    req.shard = shard;

    if (!ctx()) {
      Log_fatal("get_remote() failed: helper() undefined.");
    }

    int peer = worker_for_shard(shard);

//    Log_debug("Sending get request to: (%d, %d)", peer, shard);
    workers[peer]->get(req, &resp);

    if (resp.missing_key) {
      return false;
    }

    if (v != NULL) {
      *v = val::from_str<V>(resp.kv_data[0].value);
    }

    return true;
  }

  void start_checkpoint(const string& f, bool deltaOnly) {
    Log_fatal("Not implemented.");
  }

  void finish_checkpoint() {
    Log_fatal("Not implemented.");
  }

  void restore(const string& f) {
    Log_fatal("Not implemented.");
  }

  int flush() {
    int count = 0;
    TableData put;

    rpc::FutureGroup g;

    for (size_t i = 0; i < shards_.size(); ++i) {
      if (!is_local_shard(i)) {
        put.kv_data.clear();

        {
          GRAB_LOCK;
          ShardT<K, V>* t = (ShardT<K, V>*) shards_[i];
          for (auto j : *t) {
            put.kv_data.push_back( {val::to_str(j.first), val::to_str(j.second)});
          }
          t->clear();
        }

        if (put.kv_data.empty()) {
          continue;
        }

        put.shard = i;
        put.source = ctx()->id();
        put.table = id();
        put.done = true;

        count += put.kv_data.size();
        int target = worker_for_shard(i);
        Log_debug("Writing from %d to %d", ctx()->id(), target);
        g.add(workers[target]->async_put(put));
      }
    }

    pending_writes_ = 0;
    return count;
  }

protected:
  rpc::Mutex* mutex() {
    return &m_;
  }
};

template<class K, class V>
TypeRegistry<Table>::Helper<TableT<K, V>> TableT<K, V>::register_me_;

typedef std::map<int, Table*> TableMap;

}

#include "table-inl.h"

#endif
