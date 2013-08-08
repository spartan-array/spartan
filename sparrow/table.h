#ifndef SPARROW_TABLE_H
#define SPARROW_TABLE_H

#include <map>
#include <queue>

#include <boost/functional/hash.hpp>
#include <boost/smart_ptr.hpp>
#include <boost/thread.hpp>
#include <boost/tuple/tuple.hpp>
#include <boost/unordered_map.hpp>

#include "glog/logging.h"

#include "sparrow/util/common.h"
#include "sparrow/util/file.h"
#include "sparrow/util/marshal.h"
#include "sparrow/util/registry.h"
#include "sparrow/util/rpc.h"
#include "sparrow/util/timer.h"

#include "sparrow/sparrow.pb.h"

namespace sparrow {

using std::make_pair;
using boost::make_tuple;

// How many entries to prefetch for remote iterators.
// TODO(power) -- this should be changed to always fetch
// X MB.
static const int kDefaultIteratorFetch = 2048;

// Flush changes after this writes.
static const int kDefaultFlushFrequency = 1000000;

#define GLOBAL_TABLE_USE_SCOPEDLOCK 0

#if GLOBAL_TABLE_USE_SCOPEDLOCK == 0
#define GRAB_LOCK do { } while(0)
#else
#define GRAB_LOCK boost::recursive_mutex::scoped_lock sl(mutex())
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

class Initable {
public:
  virtual ~Initable() {

  }
  virtual void init(const std::string& opts) = 0;
  virtual int type_id() = 0;
};

class Sharder: public Initable {
public:
};

class Accumulator: public Initable {
public:
};

class Selector: public Initable {
public:
};

template<class T>
class AccumulatorT: public Accumulator {
public:
  virtual ~AccumulatorT() {
  }
  virtual void accumulate(T* v, const T& update) const = 0;
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

// This interface is used by tables to communicate with the outside
// world and determine the current state of a computation.
struct TableContext {
  virtual ~TableContext() {
  }
  virtual int id() const = 0;
  virtual int epoch() const = 0;
  virtual int peer_for_shard(int table, int shard) const = 0;
  virtual void flush_network() = 0;
};

class TableIterator {
private:
public:
  virtual ~TableIterator() {
  }
  virtual std::string key_str() = 0;
  virtual std::string value_str() = 0;
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
  virtual void start_checkpoint(const string& f, bool delta) = 0;
  virtual void finish_checkpoint() = 0;
  virtual void restore(const string& f) = 0;
  virtual void write_delta(const TableData& put) = 0;
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
  Sharder *sharder;
  Accumulator *accum;
  Selector *selector;

  int flush_frequency;

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
    return shard_info_[shard].tainted();
  }

  int worker_for_shard(int shard) const {
    return shard_info_[shard].owner();
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
      return shard_info_[shard].entries();
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
  virtual std::string get_str(const std::string&) = 0;
  virtual bool contains_str(const std::string&) = 0;
  virtual void update_str(const std::string&, const std::string&) = 0;
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

private:
  Table* table_;
  IteratorRequest request_;
  IteratorResponse response_;

  int shard_;
  int index_;
  bool done_;

  std::queue<std::pair<std::string, std::string> > cached_results;
  size_t fetch_num_;
};

template<class K, class V>
class ShardT: public Shard {
private:
  typedef boost::unordered_map<K, V> Map;
  Map data_;
public:
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

  bool empty() const {
    return data_.empty();
  }

  void clear() {
    data_.clear();
  }

  V& operator[](const K& k) {
    return data_[k];
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
public:
  LocalIterator(ShardT<K, V>& m) :
      begin_(m.begin()), cur_(m.begin()), end_(m.end()) {

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
  boost::recursive_mutex m_;
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
    flush_frequency = kDefaultFlushFrequency;

    shards_.resize(num_shards);
    for (int i = 0; i < num_shards; ++i) {
      shards_[i] = new ShardT<K, V>;
    }

    shard_info_.resize(num_shards);
  }

  virtual ~TableT() {
    for (auto p : shards_) {
      delete p;
    }
  }

  bool is_local_key(const K& key) {
    return is_local_shard(shard_for_key(key));
  }

  int shard_for_key(const K& k) {
    return ((SharderT<K>*) sharder)->shard_for_key(k, this->num_shards());
  }

  ShardT<K, V>& typed_shard(int id) {
    return *((ShardT<K, V>*) this->shards_[id]);
  }

  V get_local(const K& k) {
    int shard = this->shard_for_key(k);
    CHECK(is_local_shard(shard)) << " non-local for shard: " << shard;
    return typed_shard(shard)[k];
  }

  void update(const K& k, const V& v) {
    int shard_id = this->shard_for_key(k);

    GRAB_LOCK;;

    ShardT<K, V>& s = typed_shard(shard_id);
    typename ShardT<K, V>::iterator i = s.find(k);
    if (i == s.end()) {
      s[k] = v;
    } else {
      ((AccumulatorT<K>*) accum)->accumulate(&i->second, v);
    }

    ++pending_writes_;
    if (pending_writes_ > flush_frequency) {
      flush();
      this->handle_put_requests();
    }
  }

  bool _get(const K& k, V* v) {
    int shard = this->shard_for_key(k);

    if (tainted(shard)) {
      GRAB_LOCK;
      while (tainted(shard)) {
        this->handle_put_requests();
        sched_yield();
      }
    }

    if (is_local_shard(shard)) {
      ShardT<K, V>& s = (ShardT<K, V>&) (*shards_[shard]);
      typename ShardT<K, V>::iterator i = s.find(k);
      if (i == s.end()) {
        return false;
      }

      if (v != NULL) {
        if (selector != NULL) {
          *v = ((SelectorT<K, V>*)selector)->select(k, i->second);
        } else {
          *v = i->second;
        }
      }

      return true;
    }

    // Send any pending updates before trying to do a fetch.
    // We could alternatively try and patch up the remote
    // value with our local updates.
    flush();

    return get_remote(shard, k, v);
  }

  V get(const K& k) {
    V out;
    _get(k, &out);
    return out;
  }

  bool contains(const K& k) {
    return _get(k, NULL);
  }

  void remove(const K& k) {
    LOG(FATAL)<< "Not implemented!";
  }

  // Untyped operations:
  bool contains_str(const std::string& k) {
    return contains(val::from_str<K>(k));
  }

  std::string get_str(const std::string& k) {
    return val::to_str(get(val::from_str<K>(k)));
  }

  void update_str(const std::string& k, const std::string& v) {
    update(val::from_str<K>(k), val::from_str<V>(v));
  }

  Shard* create_local(int shard_id) {
    return new ShardT<K, V>();
  }

  TypedIterator<K, V>* get_iterator() {
    vector<TypedIterator<K, V>*> iters;
    for (int i = 0; i < num_shards(); ++i) {
      iters.push_back(get_iterator(i));
    }
    return new MergedIterator<K, V>(iters);
  }

  TypedIterator<K, V>* get_iterator(int shard) {
    if (this->is_local_shard(shard)) {
      return new LocalIterator<K, V>((ShardT<K, V>&) *(shards_[shard]));
    } else {
      return new RemoteIterator<K, V>(this, shard);
    }
  }

  void update_partitions(const PartitionInfo& info) {
    shard_info_[info.shard()].CopyFrom(info);
  }

  bool is_local_shard(int shard) {
    if (!ctx()) return false;
    return worker_for_shard(shard) == ctx()->id();
  }

  bool get_remote(int shard, const K& k, V* v) {
    {
      boost::recursive_mutex::scoped_lock sl(mutex());
      if (cache_.find(k) != cache_.end()) {
        CacheEntry& c = cache_[k];
        *v = c.val;
        return true;
      }
    }

    HashGet req;
    TableData resp;

    req.set_key(val::to_str(k));
    req.set_table(id());
    req.set_shard(shard);

    if (!ctx()) {
      LOG(FATAL) << "get_remote() failed: helper() undefined.";
    }
    int peer = ctx()->peer_for_shard(id(), shard);

    DCHECK_GE(peer, 0);
    DCHECK_LT(peer, rpc::NetworkThread::Get()->size() - 1);

    VLOG(2) << "Sending get request to: " << make_pair(peer, shard);
    rpc::NetworkThread::Get()->Call(peer + 1, MessageTypes::GET, req, &resp);

    if (resp.missing_key()) {
      return false;
    }

    if (v != NULL) {
      *v = val::from_str<V>(resp.kv_data(0).value());
    }

    boost::recursive_mutex::scoped_lock sl(mutex());
    CacheEntry c = {Now(), *v};
    cache_[k] = c;
    return true;
  }

  void clear() {
    ClearTable req;

    req.set_table(this->id());
    VLOG(2) << StringPrintf("Sending clear request (%d)", req.table());

    rpc::NetworkThread::Get()->SyncBroadcast(MessageTypes::CLEAR_TABLE, req);
  }

  void start_checkpoint(const string& f, bool deltaOnly) {
    LOG(FATAL) << "Not implemented.";
  }

  void finish_checkpoint() {
    LOG(FATAL) << "Not implemented.";
  }

  void write_delta(const TableData& d) {
    LOG(FATAL) << "Not implemented.";
  }

  void restore(const string& f) {
    LOG(FATAL) << "Not implemented.";
  }

  void handle_put_requests() {
    ctx()->flush_network();
  }

  int flush() {
    int count = 0;

    TableData put;
    boost::recursive_mutex::scoped_lock sl(mutex());
    for (size_t i = 0; i < shards_.size(); ++i) {
      ShardT<K, V>* t = (ShardT<K, V>*) shards_[i];
      if (!is_local_shard(i) && (shard_info_[i].dirty() || !t->empty())) {
        // Always send at least one chunk, to ensure that we clear taint on
        // tables we own.
        put.Clear();

        for (auto j : *t) {
          KV* put_kv = put.add_kv_data();
          put_kv->set_key(val::to_str(j.first));
          put_kv->set_value(val::to_str(j.second));
        }
        t->clear();

        put.set_shard(i);
        put.set_source(ctx()->id());
        put.set_table(id());
        put.set_epoch(ctx()->epoch());

        put.set_done(true);

        count += put.kv_data_size();
        rpc::NetworkThread::Get()->Send(worker_for_shard(i) + 1,
            MessageTypes::PUT_REQUEST, put);
      }
    }

    if (count > 0) {
      ctx()->flush_network();
    }

    pending_writes_ = 0;
    return count;
  }

protected:
  boost::recursive_mutex& mutex() {
    return m_;
  }
};

#ifndef SWIG
template<class K, class V>
TypeRegistry<Table>::Helper<TableT<K, V>> TableT<K, V>::register_me_;
#endif

typedef std::map<int, Table*> TableMap;

}

#include "table-inl.h"

#endif
