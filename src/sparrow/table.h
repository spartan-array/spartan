#ifndef SPARROW_TABLE_H
#define SPARROW_TABLE_H

#include "sparrow/util/common.h"
#include "sparrow/util/file.h"
#include "sparrow/util/marshal.h"
#include "sparrow/util/registry.h"

#include <map>
#include <queue>

#include <boost/functional/hash.hpp>
#include <boost/smart_ptr.hpp>
#include <boost/thread.hpp>
#include <boost/unordered_map.hpp>

#include "sparrow/sparrow.pb.h"

namespace sparrow {

// How many entries to prefetch for remote iterators.
// TODO(power) -- this should be changed to always fetch
// X MB.
static const int kDefaultIteratorFetch = 2048;

// Flush changes after this writes.
static const int kDefaultFlushFrequency = 1000000;

typedef std::string TableKey;
typedef std::string TableValue;

template <class T>
std::string prim_to_string(const T& t) {
  std::string result;
  result.resize(sizeof(t));
  *((T*)result.data()) = t;
  return result;
}

template <class T>
T string_to_prim(const std::string& t) {
  return *((T*)t.data());
}

struct Accumulator {
  virtual ~Accumulator() {
  }
  virtual void accumulate(TableValue* current,
      const TableValue& update) const = 0;

  // name() should match the name this accumulator was registered as.
  virtual const char* name() const = 0;
};

#define REGISTER_ACCUMULATOR(name, klass)\
    static TypeRegistry<Accumulator>::Helper<klass > k_helper_ ## __FILE__ ## __LINE__ (name);

struct Sharder {
  virtual ~Sharder() {
  }
  virtual int shard_for_key(const TableKey& k, int num_shards) const = 0;
};

// This interface is used by tables to communicate with the outside
// world and determine the current state of a computation.
struct TableHelper {
  virtual ~TableHelper() {}
  virtual int id() const = 0;
  virtual int epoch() const = 0;
  virtual int peer_for_shard(int table, int shard) const = 0;
  virtual void check_network() = 0;
};

struct Sharding {
  struct Modulo: public Sharder {
    int shard_for_key(const TableKey& key, int num_shards) const {
      return boost::hash_value(key) % num_shards;
    }
  };
};

class TableIterator {
private:
public:
  virtual ~TableIterator() {
  }
  virtual const TableKey& key() = 0;
  virtual const TableValue& value() = 0;
  virtual bool done() = 0;
  virtual void next() = 0;
};

class Checkpointable {
public:
  virtual ~Checkpointable() {}
  virtual void start_checkpoint(const string& f, bool delta) = 0;
  virtual void finish_checkpoint() = 0;
  virtual void restore(const string& f) = 0;
  virtual void write_delta(const TableData& put) = 0;
};

class Shard: public boost::unordered_map<TableKey, TableValue> {
};

class Table {
private:
  struct CacheEntry {
    double last_read_time;
    TableValue val;
  };
  boost::unordered_map<TableKey, CacheEntry> cache_;

  int pending_writes_;
  int id_;
  TableHelper *helper_;
  std::vector<Shard*> shards_;
  std::vector<PartitionInfo> shard_info_;

  boost::recursive_mutex m_;
public:

  Sharder* sharder;
  Accumulator* accum;

  int flush_frequency;

  Table(int id, int num_shards) {
    id_ = id;
    sharder = new Sharding::Modulo();
    pending_writes_ = 0;
    helper_ = NULL;
    flush_frequency = kDefaultFlushFrequency;

    initialize(num_shards);
  }

  void initialize(int num_shards) {
    shards_.resize(num_shards);
    for (int i = 0; i < num_shards; ++i) {
      shards_[i] = new Shard;
    }

    shard_info_.resize(num_shards);
  }

  virtual ~Table();

  bool is_local_shard(int shard);
  bool is_local_key(const TableKey& key) {
    return is_local_shard(shard_for_key(key));
  }

  int64_t shard_size(int shard);

// Fill in a response from a remote worker for the given key
  bool tainted(int shard) {
    return shard_info_[shard].tainted();
  }

  int worker_for_shard(int shard) const {
    return shard_info_[shard].owner();
  }

  bool is_local_shard(int shard) const {
    return worker_for_shard(shard) == helper()->id();
  }

  int num_shards() const {
    return shards_.size();
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

  int shard_for_key(const TableKey& k);

  void put(const TableKey& k, const TableValue& v);
  void update(const TableKey& k, const TableValue& v);

  const TableValue& get(const TableKey& k);
  bool contains(const TableKey& k);
  void remove(const TableKey& k);
  void clear();

  const TableValue& get_local(const TableKey& k);
  bool get_remote(int shard, const TableKey& k, TableValue *v);

  void start_checkpoint(const string& f, bool deltaOnly);
  void finish_checkpoint();
  void write_delta(const TableData& d);
  void restore(const string& f);

  TableIterator* get_iterator(int shard);

  void update_partitions(const PartitionInfo& sinfo);
  int send_updates();
  void handle_put_requests();
  int pending_writes();

  void set_helper(TableHelper* h) {
    helper_ = h;
  }

  TableHelper* helper() const {
    return helper_;
  }

protected:
  boost::recursive_mutex& mutex() {
    return m_;
  }
  virtual Shard* create_local(int shard_id);
};

typedef std::map<int, Table*> TableMap;

}

#endif
