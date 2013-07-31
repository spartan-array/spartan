#ifndef ACCUMULATOR_H
#define ACCUMULATOR_H

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

static const int kDefaultIteratorFetch = 2048;

class HashGet;
class TableData;
class TableDescriptor;

typedef std::string TableKey;
typedef std::string TableValue;

template<class T>
bool read(T* v, StringPiece src) {
  StringReader r(src);
  return v->read(&r);
}

template<class T>
void write(const T& v, string* out) {
  StringWriter w(out);
  v.write(&w);
}

struct Accumulator {
  virtual ~Accumulator() {
  }
  virtual void accumulate(TableValue* current,
      const TableValue& update) const = 0;
};

struct Sharder {
  virtual ~Sharder() {
  }
  virtual int shard_for_key(const TableKey& k, int num_shards) const = 0;
};

// This interface is used by global tables to communicate with the outside
// world and determine the current state of a computation.
struct TableHelper {
  virtual ~TableHelper() {

  }

  virtual int id() const = 0;
  virtual int epoch() const = 0;
  virtual int peer_for_shard(int table, int shard) const = 0;
  virtual void handle_put_request() = 0;
};

template<class V>
struct AccumulatorT {
  virtual ~AccumulatorT() {
  }
  virtual void accumulate(V* current, const V& update) const = 0;

  void accumulate(TableValue* current, const TableValue& update) {
    //accumulate(current, TableValueT<V>::get(update));
    LOG(FATAL)<< "Not implemented.";
  }
};

template<class V>
struct Accumulators {
  struct Min: public AccumulatorT<V> {
    virtual void accumulate(V* current, const V& update) const {
      *current = std::min(*current, update);
    }
  };

  struct Max: public AccumulatorT<V> {
    virtual void accumulate(V* current, const V& update) const {
      *current = std::max(*current, update);
    }
  };

  struct Sum: public AccumulatorT<V> {
    virtual void accumulate(V* current, const V& update) const {
      *current += update;
    }
  };

  struct Replace: public AccumulatorT<V> {
    virtual void accumulate(V* current, const V& update) const {
      *current = update;
    }
  };

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

class PartitionInfo;

class Checkpointable {
public:
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

  Sharder* sharder_;
  std::vector<Shard*> partitions_;
  std::vector<PartitionInfo> partinfo_;

  boost::recursive_mutex m_;

public:
  Table(int id, int num_shards) {
    id_ = id;
    sharder_ = new Sharding::Modulo();
    pending_writes_ = 0;
    helper_ = NULL;

    initialize(num_shards);
  }

  void initialize(int num_shards) {
    partitions_.resize(num_shards);
    for (int i = 0; i < num_shards; ++i) {
      partitions_[i] = new Shard;
    }

    partinfo_.resize(num_shards);
  }

  virtual ~Table();

  bool is_local_shard(int shard);
  bool is_local_key(const TableKey& key) {
    return is_local_shard(shard_for_key(key));
  }

  int64_t shard_size(int shard);

// Fill in a response from a remote worker for the given key.
  void handle_get(const HashGet& req, TableData* resp);

  PartitionInfo* get_partition_info(int shard) {
    return &partinfo_[shard];
  }

  Shard& get_partition(int shard) {
    return *partitions_[shard];
  }

  bool tainted(int shard) {
    return partinfo_[shard].tainted();
  }

  int worker_for_shard(int shard) const {
    return partinfo_[shard].owner();
  }

  bool is_local_shard(int shard) const {
    return worker_for_shard(shard) == helper()->id();
  }

  int num_shards() const {
    return partitions_.size();
  }

  int id() const {
    return id_;
  }

  Shard* shard(int id) {
    return partitions_[id];
  }

  int shard_for_key(const TableKey& k);

  void put(const TableKey& k, const TableValue& v);
  void update(const TableKey& k, const TableValue& v, Accumulator* accum);

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
  void apply_updates(const sparrow::TableData& req);
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

static const int kWriteFlushCount = 1000000;


}

#endif
