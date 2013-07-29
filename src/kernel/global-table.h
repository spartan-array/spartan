#ifndef GLOBALTABLE_H_
#define GLOBALTABLE_H_

#include "kernel/shard.h"

#include "util/file.h"
#include "util/marshal.h"
#include "util/rpc.h"
#include "util/timer.h"
#include "util/tuple.h"

#include <queue>
#include <tr1/unordered_map>

#include "kernel/table.h"
#include <boost/thread/recursive_mutex.hpp>

namespace sparrow {

class Worker;
class Master;

#define GLOBAL_TABLE_USE_SCOPEDLOCK 0

#if GLOBAL_TABLE_USE_SCOPEDLOCK == 0
#define GRAB_LOCK do { } while(0)
#else
#define GRAB_LOCK boost::recursive_mutex::scoped_lock sl(mutex())
#endif

class TableDataReader: public Reader {

};

class TableDataWriter: public Writer {

};

class RemoteIterator: public TableIterator {
public:
  RemoteIterator(Table *table, int shard, uint32_t fetch_num =
      kDefaultIteratorFetch);

  bool done();
  void next();

  const TableValue& key() {
    key_->read(cached_results.front().first);
    return *key_;
  }

  const TableValue& value() {
    value_->read(cached_results.front().first);
    return *value_;
  }

private:
  Table* table_;
  IteratorRequest request_;
  IteratorResponse response_;

  int shard_;
  int index_;
  TableValue::ScopedPtr key_;
  TableValue::ScopedPtr value_;
  bool done_;

  std::queue<std::pair<string, string> > cached_results;
  size_t fetch_num_;
};

template<class Key, class Value>
class GlobalTable: public Table {
public:
  virtual ~GlobalTable();

  virtual bool is_local_shard(int shard);
  virtual bool is_local_key(const Key& key);

  int64_t shard_size(int shard);

// Fill in a response from a remote worker for the given key.
  void handle_get(const HashGet& req, TableData* resp);

  PartitionInfo* get_partition_info(int shard) {
    return &partinfo_[shard];
  }

  HashShard<Key, Value>* get_partition(int shard) {
    return partitions_[shard];
  }

  bool tainted(int shard) {
    return get_partition_info(shard)->tainted;
  }

  int worker_for_shard(int shard) {
    return get_partition_info(shard)->sinfo.owner();
  }

  TableValue* new_key();
  TableValue* new_value();

  int get_shard(const Key& k);

  void put(const Key& k, const Value& v);
  void update(const Key& k, const Value& v, Accumulator* accum);

  Value get(const Key& k);
  bool contains(const Key& k);
  void remove(const Key& k);
  void clear();

  const Value& get_local(const Key& k);
  bool get_remote(int shard, const Key& k, Value* v);

  void start_checkpoint(const string& f, bool deltaOnly);
  void finish_checkpoint();
  void write_delta(const TableData& d);
  void restore(const string& f);

  TableIterator* get_iterator(int shard, size_t fetch_num =
      kDefaultIteratorFetch);

  HashShard<Key, Value>* partition(int idx) {
    return partitions_[idx];
  }

  void update_partitions(const ShardInfo& sinfo);
  void apply_updates(const sparrow::TableData& req);
  int send_updates();
  void handle_put_requests();
  int pending_writes();

  void set_helper(TableHelper*);
  TableHelper* helper();

protected:
  int worker_id_;

  Sharder* sharder_;
  std::vector<HashShard<Key, Value>*> partitions_;
  std::vector<HashShard<Key, Value>*> cache_;

  std::vector<TableData*> writebufs_;
  std::vector<PartitionInfo> partinfo_;

  boost::recursive_mutex m_;

  int pending_writes_;

  boost::recursive_mutex& mutex() {
    return m_;
  }

  struct CacheEntry {
    double last_read_time;
    Value val;
  };

  std::tr1::unordered_map<Key, CacheEntry> remote_cache_;

  virtual HashShard<Key, Value>* create_local(int shard_id);
};

static const int kWriteFlushCount = 1000000;

template<class Key, class Value>
int GlobalTable<Key, Value>::get_shard(const Key& k) {
  return this->sharder_->shard_for_key(k, this->num_shards());
}

template<class Key, class Value>
const Value& GlobalTable<Key, Value>::get_local(const Key& k) {
  int shard = this->get_shard(k);
  CHECK(is_local_shard(shard)) << " non-local for shard: " << shard;
  return partition(shard)->get(k);
}

template<class Key, class Value>
void GlobalTable<Key, Value>::put(const Key& k, const Value& v) {
  LOG(FATAL)<< "Need to implement.";
  int shard = this->get_shard(k);

  GRAB_LOCK;
  partition(shard)->put(k, v);

  if (!is_local_shard(shard)) {
    ++pending_writes_;
  }

  if (pending_writes_ > kWriteFlushCount) {
    send_updates();
  }

  PERIODIC(0.1, {this->handle_put_requests();});
}

template<class Key, class Value>
void GlobalTable<Key, Value>::update(const Key& k, const Value& v,
    Accumulator* accum) {
  int shard = this->get_shard(k);

  GRAB_LOCK;;

  if (is_local_shard(shard)) {
    partition(shard)->update(k, v, accum);
  } else {
    LOG(FATAL)<< "Not implemented.";
  }

  ++pending_writes_;
  if (pending_writes_ > kWriteFlushCount) {
    send_updates();
  }

  PERIODIC(0.1, {this->handle_put_requests();});
}

template<class Key, class Value>
Value GlobalTable<Key, Value>::get(const Key& k) {
  int shard = this->get_shard(k);

  // If we received a request for this shard; but we haven't received all of the
  // data for it yet. Continue reading from other workers until we do.
  // New for triggers: be sure to not recursively apply updates.
  if (tainted(shard)) {
    GRAB_LOCK;
    while (tainted(shard)) {
      this->handle_put_requests();
      sched_yield();
    }
  }

  PERIODIC(0.1, this->handle_put_requests());

  if (is_local_shard(shard)) {
    GRAB_LOCK;
    return partition(shard)->get(k);
  }

  Value v;
  get_remote(shard, k, &v);
  return v;
}

template<class Key, class Value>
bool GlobalTable<Key, Value>::contains(const Key& k) {
  int shard = this->get_shard(k);

  // If we received a request for this shard; but we haven't received all of the
  // data for it yet. Continue reading from other workers until we do.
  // New for triggers: be sure to not recursively apply updates.
  if (tainted(shard)) {
    GRAB_LOCK;
    while (tainted(shard)) {
      this->handle_put_requests();
      sched_yield();
    }
  }

  if (is_local_shard(shard)) {
    return partition(shard)->contains(k);
  }

  Value v;
  return get_remote(shard, k, &v);
}

template<class Key, class Value>
void GlobalTable<Key, Value>::remove(const Key& k) {
  LOG(FATAL)<< "Not implemented!";
}

template<class Key, class Value>
HashShard<Key, Value>* GlobalTable<Key, Value>::create_local(int shard_id) {
  return new HashShard<Key, Value>(shard_id);
}

template<class Key, class Value>
TableIterator* GlobalTable<Key, Value>::get_iterator(int shard,
    size_t fetch_num) {
  if (this->is_local_shard(shard)) {
    return partitions_[shard]->get_iterator();
  } else {
    return new RemoteIterator(this, shard, fetch_num);
  }
}

template<class Key, class Value>
void GlobalTable<Key, Value>::update_partitions(const ShardInfo& info) {
  partinfo_[info.shard()].sinfo.CopyFrom(info);
}

template<class Key, class Value>
GlobalTable<Key, Value>::~GlobalTable() {
  for (int i = 0; i < partitions_.size(); ++i) {
    delete partitions_[i];
    delete writebufs_[i];
  }
}

template<class Key, class Value>
bool GlobalTable<Key, Value>::is_local_shard(int shard) {
  if (!helper()) return false;
  return worker_for_shard(shard) == helper()->id();
}

template<class Key, class Value>
int64_t GlobalTable<Key, Value>::shard_size(int shard) {
  if (is_local_shard(shard)) {
    return partitions_[shard]->size();
  } else {
    return partinfo_[shard].sinfo.entries();
  }
}

template<class Key, class Value>
bool GlobalTable<Key, Value>::get_remote(int shard, const Key& k, Value* v) {
  {
    VLOG(3) << "Entering get_remote";
    boost::recursive_mutex::scoped_lock sl(mutex());
    VLOG(3) << "Entering get_remote and locked";
    if (remote_cache_.find(k) != remote_cache_.end()) {
      CacheEntry& c = remote_cache_[k];
      *v = c.val;
      return true;
    }
  }

  HashGet req;
  TableData resp;

  req.set_key(k.to_str());
  req.set_table(id());
  req.set_shard(shard);

  if (!helper()) {
    LOG(FATAL)<< "get_remote() failed: helper() undefined.";
  }
  int peer = helper()->peer_for_shard(id(), shard);

  DCHECK_GE(peer, 0);
  DCHECK_LT(peer, rpc::NetworkThread::Get()->size() - 1);

  VLOG(2) << "Sending get request to: " << MP(peer, shard);
  rpc::NetworkThread::Get()->Call(peer + 1, MTYPE_GET, req, &resp);

  if (resp.missing_key()) {
    return false;
  }

  v->read(resp.kv_data(0).value());

  boost::recursive_mutex::scoped_lock sl(mutex());
  CacheEntry c = { Now(), *v };
  remote_cache_[k] = c;
  return true;
}

template<class Key, class Value>
void GlobalTable<Key, Value>::clear() {
  ClearTable req;

  req.set_table(this->id());
  VLOG(2) << StringPrintf("Sending clear request (%d)", req.table());

  rpc::NetworkThread::Get()->SyncBroadcast(MTYPE_CLEAR_TABLE, req);
}

template<class Key, class Value>
void GlobalTable<Key, Value>::start_checkpoint(const string& f,
    bool deltaOnly) {
  for (int i = 0; i < partitions_.size(); ++i) {
    HashShard<Key, Value>* t = partitions_[i];

    if (is_local_shard(i)) {
      t->start_checkpoint(
          f + StringPrintf(".%05d-of-%05d", i, partitions_.size()), deltaOnly);
    }
  }
}

template<class Key, class Value>
void GlobalTable<Key, Value>::finish_checkpoint() {
  for (int i = 0; i < partitions_.size(); ++i) {
    HashShard<Key, Value>* t = partitions_[i];

    if (is_local_shard(i)) {
      t->finish_checkpoint();
    }
  }
}

template<class Key, class Value>
void GlobalTable<Key, Value>::write_delta(const TableData& d) {
  if (!is_local_shard(d.shard())) {
    LOG_EVERY_N(INFO, 1000) << "Ignoring delta write for forwarded data";
    return;
  }

  partitions_[d.shard()]->write_delta(d);
}

template<class Key, class Value>
void GlobalTable<Key, Value>::restore(const string& f) {
  for (int i = 0; i < partitions_.size(); ++i) {
    HashShard<Key, Value>* t = partitions_[i];

    if (is_local_shard(i)) {
      t->restore(f + StringPrintf(".%05d-of-%05d", i, partitions_.size()));
    } else {
      t->clear();
    }
  }
}

template<class Key, class Value>
void GlobalTable<Key, Value>::handle_put_requests() {
  helper()->handle_put_request();
}

template<class Key, class Value>
int GlobalTable<Key, Value>::send_updates() {
  int count = 0;

  TableData put;
  boost::recursive_mutex::scoped_lock sl(mutex());
  for (int i = 0; i < partitions_.size(); ++i) {
    HashShard<Key, Value>* t = partitions_[i];
    if (!is_local_shard(i) && (get_partition_info(i)->dirty || !t->empty())) {
      // Always send at least one chunk, to ensure that we clear taint on
      // tables we own.
      do {
        put.Clear();

        VLOG(3) << "Sending update from non-trigger table ";
        LOG(FATAL)<< "TODO: serialize table.";
        t->clear();

        put.set_shard(i);
        put.set_source(helper()->id());
        put.set_table(id());
        put.set_epoch(helper()->epoch());

        put.set_done(true);

        count += put.kv_data_size();
        rpc::NetworkThread::Get()->Send(worker_for_shard(i) + 1,
            MTYPE_PUT_REQUEST, put);
      } while (!t->empty());

      t->clear();
    }
  }

  pending_writes_ = 0;
  return count;
}

template<class Key, class Value>
int GlobalTable<Key, Value>::pending_writes() {
  int64_t s = 0;
  for (int i = 0; i < partitions_.size(); ++i) {
    HashShard<Key, Value> *t = partitions_[i];
    if (!is_local_shard(i)) {
      s += t->size();
    }
  }

  return s;
}

}

#endif /* GLOBALTABLE_H_ */
