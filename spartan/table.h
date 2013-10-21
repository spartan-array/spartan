#ifndef SPARTAN_TABLE_H
#define SPARTAN_TABLE_H

#include <map>
#include <queue>

#include <boost/functional/hash.hpp>
#include <boost/smart_ptr.hpp>
#include <boost/tuple/tuple.hpp>
#include <boost/unordered_map.hpp>

#include "spartan/util/common.h"
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

class Table;
class Master;

#define GLOBAL_TABLE_USE_SCOPEDLOCK 1

#if GLOBAL_TABLE_USE_SCOPEDLOCK == 0
#define GRAB_LOCK do { } while(0)
#else
#define GRAB_LOCK rpc::ScopedLock sl(mutex())
#endif

class Accumulator: public Initable {
public:
  virtual void accumulate(const RefPtr& key, RefPtr* v,
      const RefPtr& update) const = 0;
};

class Sharder: public Initable {
public:
  virtual size_t shard_for_key(const RefPtr& k, int num_shards) const = 0;
};

class Selector: public Initable {
public:
  virtual RefPtr select(const RefPtr& k, const RefPtr& v) = 0;
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
  virtual RefPtr key() = 0;
  virtual RefPtr value() = 0;
  virtual int shard() = 0;
  virtual bool done() = 0;
  virtual void next() = 0;
};

class Shard {
private:
  typedef boost::unordered_map<RefPtr, RefPtr> Map;
  Map data_;
public:
  typedef typename Map::iterator iterator;
  iterator begin() {
    return data_.begin();
  }

  iterator end() {
    return data_.end();
  }

  iterator find(const RefPtr& k) {
    return data_.find(k);
  }

  void insert(const RefPtr& k, const RefPtr& v) {
    CHECK_NE(k.get(), NULL);
    data_[k] = v;
    // data_.insert(make_pair(k, v));
  }

  bool empty() const {
    return data_.empty();
  }

  void clear() {
    data_.clear();
  }

  const RefPtr& get(const RefPtr& k) {
    return data_.find(k)->second;
  }

  size_t size() const {
    return data_.size();
  }
};


class RemoteIterator: public TableIterator {
public:
  RemoteIterator(Table *table, int shard, uint32_t fetch_num =
      kDefaultIteratorFetch);

  bool done();
  void next();
  RefPtr key();
  RefPtr value();

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

class LocalIterator: public TableIterator {
private:
  typename Shard::iterator begin_;
  typename Shard::iterator cur_;
  typename Shard::iterator end_;

  int shard_;
public:
  LocalIterator(int shard, Shard& m) :
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

  RefPtr key() {
    return cur_->first;
  }

  RefPtr value() {
    return cur_->second;
  }
};

class MergedIterator: public TableIterator {
private:
  typedef std::vector<TableIterator*> IterList;
  IterList iters_;

  TableIterator* cur() {
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

  RefPtr key() {
    return cur()->key();
  }

  RefPtr value() {
    return cur()->value();
  }
};


class Table {
public:
  struct CacheEntry {
    double last_read_time;
    RefPtr val;
  };

  std::vector<WorkerProxy*> workers;
  Sharder *sharder;
  Accumulator *combiner;
  Accumulator *reducer;
  Selector *selector;

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

  Table() {
    pending_updates_ = 0;
    sharder = NULL;
    ctx_ = NULL;
    id_ = -1;
    combiner = reducer = NULL;
    selector = NULL;
  }

  virtual ~Table() {
    //Log_info("Deleting table %d", id());
    for (auto p : shards_) {
      delete p;
    }
  }

  void init(int id, int num_shards) {
    id_ = id;

    shards_.resize(num_shards);
    shard_info_.resize(num_shards);
    for (int i = 0; i < num_shards; ++i) {
      shards_[i] = new Shard;
      shard_info_[i].owner = -1;
      shard_info_[i].shard = i;
      shard_info_[i].table = id;
    }
  }

  bool is_local_key(const RefPtr& key) {
    return is_local_shard(shard_for_key(key));
  }

  int shard_for_key(const RefPtr& k) {
    CHECK_NE(sharder, NULL);
    return sharder->shard_for_key(k, this->num_shards());
  }

  Shard& typed_shard(int id) {
    return *((Shard*) this->shards_[id]);
  }

  void update(int shard, const RefPtr& k, const RefPtr& v);

  bool _get(int shard, const RefPtr& k, RefPtr* v);

  RefPtr get(int shard, const RefPtr& k) {
    RefPtr out;
    _get(shard, k, &out);
    return out;
  }

  bool contains(int shard, const RefPtr& k) {
    return _get(shard, k, NULL);
  }

  void remove(const RefPtr& k) {
    Log_fatal("Not implemented!");
  }

  Shard* create_local(int shard_id) {
    return new Shard();
  }

  TableIterator* get_iterator() {
    std::vector<TableIterator*> iters;
    for (int i = 0; i < num_shards(); ++i) {
      iters.push_back(get_iterator(i));
    }
    return new MergedIterator(iters);
  }

  TableIterator* get_iterator(int shard) {
    if (this->is_local_shard(shard)) {
      auto s = shards_[shard];
      return new LocalIterator(shard, *((Shard*) s));
    } else {
      return new RemoteIterator(this, shard);
    }
  }

  void update_partitions(const PartitionInfo& info) {
    shard_info_[info.shard] = info;
  }

  bool is_local_shard(int shard) {
    return worker_for_shard(shard) == ctx()->id();
  }

  bool get_remote(int shard, const RefPtr& k, RefPtr* v);

  void start_checkpoint(const std::string& f, bool deltaOnly) {
    Log_fatal("Not implemented.");
  }

  void finish_checkpoint() {
    Log_fatal("Not implemented.");
  }

  void restore(const std::string& f) {
    Log_fatal("Not implemented.");
  }

  int flush();

private:
  boost::unordered_map<RefPtr, CacheEntry> cache_;
  rpc::Mutex m_;
  int pending_updates_;

protected:
  std::vector<PartitionInfo> shard_info_;
  std::vector<Shard*> shards_;
  int id_;
  TableContext *ctx_;
  rpc::Mutex* mutex() {
    return &m_;
  }
};

typedef std::map<int, Table*> TableMap;

} // namespace spartan

#endif
