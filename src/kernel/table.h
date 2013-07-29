#ifndef ACCUMULATOR_H
#define ACCUMULATOR_H

#include "util/common.h"
#include "util/file.h"
#include "util/marshal.h"
#include "sparrow.pb.h"
#include <boost/thread.hpp>
#include <boost/dynamic_bitset.hpp>

namespace sparrow {

static const int kDefaultIteratorFetch = 2048;

struct Table;
class TableData;
class TableDescriptor;

struct Accumulator {
};

struct Sharder {
};

// This interface is used by global tables to communicate with the outside
// world and determine the current state of a computation.
struct TableHelper {
  virtual ~TableHelper();
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
};

template<class V>
struct Accumulators {
  struct Min: public Accumulator {
    virtual void accumulate(V* current, const V& update) {
      *current = std::min(*current, update);
    }
  };

  struct Max: public Accumulator {
    virtual void accumulate(V* current, const V& update) {
      *current = std::max(*current, update);
    }
  };

  struct Sum: public Accumulator {
    virtual void accumulate(V* current, const V& update) {
      *current += update;
    }
  };

  struct Replace: public Accumulator {
    virtual void accumulate(V* current, const V& update) {
      *current = update;
    }
  };

};

struct Sharding {
};

class TableValue;

template<class T>
TableValue* to_table_value(T& t);

// Interface used for serializing table values.
struct TableValue {
  typedef boost::shared_ptr<TableValue> Ptr;
  typedef boost::scoped_ptr<TableValue> ScopedPtr;

  virtual ~TableValue() {

  }

  virtual bool read(Reader* r) = 0;

  bool read(StringPiece s) {
    StringReader r(s);
    return read(&r);
  }

  virtual void write(Writer* w) const = 0;

  void write(std::string* s) const {
    StringWriter w(s);
    write(&w);
  }

  template<class T>
  static T from_str(StringPiece s) {
    T val;
    TableValue::ScopedPtr wrapper(to_table_value(val));
    wrapper->read(s);
    return val;
  }

  template<class T>
  static bool read(T* val, Reader* r) {
    TableValue::ScopedPtr wrapper(to_table_value(*val));
    return wrapper->read(r);
  }
};

class TableIterator {
private:
public:
  virtual ~TableIterator() {
  }
  virtual const TableValue& key() = 0;
  virtual const TableValue& value() = 0;
  virtual bool done() = 0;
  virtual void next() = 0;
};

class Shard {
public:
  virtual ~Shard() {

  }

  bool empty() {
    return size() == 0;
  }

  virtual int64_t size() = 0;
  virtual void clear() = 0;
  virtual void resize(int64_t size) = 0;

  virtual TableIterator* get_iterator() = 0;
  virtual int shard_id() = 0;
};

struct PartitionInfo {
  PartitionInfo() :
      dirty(false), tainted(false) {
  }
  bool dirty;
  bool tainted;
  ShardInfo sinfo;
};

struct Table {
  typedef TableIterator Iterator;

  virtual ~Table() {
  }

  virtual int id() const = 0;
  virtual int num_shards() const = 0;

  virtual void set_helper(TableHelper*) = 0;

  virtual bool is_local_shard(int) const = 0;
  virtual int worker_for_shard(int shard) const = 0;

  virtual void put(const TableValue& key, const TableValue& value) = 0;
  virtual bool get(const TableValue& key, TableValue* value) = 0;

  virtual TableValue* new_key() = 0;
  virtual TableValue* new_value() = 0;

  virtual Shard* shard(int);
  virtual int send_updates();

  virtual PartitionInfo* get_partition_info(int shard) = 0;
  virtual Iterator* get_iterator(int shard);
  virtual int pending_writes();
};

class Checkpointable {
public:
  virtual void start_checkpoint(const string& f, bool delta) = 0;
  virtual void finish_checkpoint() = 0;
  virtual void restore(const string& f) = 0;
  virtual void write_delta(const TableData& put) = 0;
};

class TableRegistry {
public:
  typedef std::map<int, Table*> Map;
  Map& tables();

  Table* table(int id);
  static TableRegistry* Get();
};

}

#endif
