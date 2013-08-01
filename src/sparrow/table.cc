#include "sparrow/table.h"
#include "sparrow/util/registry.h"
#include "sparrow/util/rpc.h"
#include "sparrow/util/timer.h"
#include "sparrow/util/tuple.h"
#include "sparrow/sparrow.pb.h"

using namespace sparrow;

#define GLOBAL_TABLE_USE_SCOPEDLOCK 0

#if GLOBAL_TABLE_USE_SCOPEDLOCK == 0
#define GRAB_LOCK do { } while(0)
#else
#define GRAB_LOCK boost::recursive_mutex::scoped_lock sl(mutex())
#endif

#define _PASTE(x, y) x ## y
#define PASTE(x, y) _PASTE(x, y)

#define MAKE_ACCUMULATOR(AccumType, ValueType)\
  struct Accum_ ## AccumType ## _ ## ValueType : public AccumType<ValueType> {\
  const char* name() const { return #ValueType #AccumType; }\
};\
static TypeRegistry<Accumulator>::Helper<Accum_ ## AccumType ## _ ## ValueType>\
    k_helper ## AccumType ## ValueType(#ValueType #AccumType)

#define MAKE_ACCUMULATORS(ValueType)\
  MAKE_ACCUMULATOR(Max, ValueType);\
  MAKE_ACCUMULATOR(Min, ValueType);\
  MAKE_ACCUMULATOR(Sum, ValueType);

namespace sparrow {

// Helper class which casts string values to struct types.
template<class V>
struct AccumulatorT: public Accumulator {
  virtual ~AccumulatorT() {
  }
  virtual void accumulate(V* current, const V& update) const = 0;

  void accumulate(TableValue* current, const TableValue& update) const {
    V* cur_val = (V*) current->data();
    const V* up_val = (const V*) update.data();

    accumulate(cur_val, *up_val);
  }
};

template<class V>
struct Min: public AccumulatorT<V> {
  virtual void accumulate(V* current, const V& update) const {
    *current = std::min(*current, update);
  }
};

template<class V>
struct Max: public AccumulatorT<V> {
  void accumulate(V* current, const V& update) const {
    *current = std::max(*current, update);
  }
};

template<class V>
struct Sum: public AccumulatorT<V> {
  void accumulate(V* current, const V& update) const {
    *current += update;
  }
};

struct Replace: public Accumulator {
  void accumulate(TableValue* current, const TableValue& update) const {
    *current = update;
  }

  const char* name() const {
    return "Replace";
  }
};

REGISTER_ACCUMULATOR("Replace", Replace);

MAKE_ACCUMULATORS(int);
MAKE_ACCUMULATORS(double);

typedef boost::unordered_map<TableKey, TableValue> Map;

class RemoteIterator: public TableIterator {
public:
  RemoteIterator(Table *table, int shard, uint32_t fetch_num =
      kDefaultIteratorFetch);

  bool done();
  void next();

  const TableKey& key();
  const TableValue& value();

private:
  Table* table_;
  IteratorRequest request_;
  IteratorResponse response_;

  int shard_;
  int index_;
  bool done_;

  std::queue<std::pair<string, string> > cached_results;
  size_t fetch_num_;
};

class LocalIterator: public TableIterator {
private:
  Map::iterator begin_;
  Map::iterator cur_;
  Map::iterator end_;
public:
  LocalIterator(Map& m) :
      begin_(m.begin()), cur_(m.begin()), end_(m.end()) {

  }

  void next() {
    ++cur_;
  }

  bool done() {
    return cur_ == end_;
  }

  const TableKey& key() {
    return cur_->first;
  }

  const TableValue& value() {
    return cur_->second;
  }
};

RemoteIterator::RemoteIterator(Table *table, int shard, uint32_t fetch_num) :
    table_(table), shard_(shard), done_(false), fetch_num_(fetch_num) {

  request_.set_table(table->id());
  request_.set_shard(shard_);
  request_.set_row_count(fetch_num_);
  int target_worker = table->worker_for_shard(shard);

  // << CRM 2011-01-18 >>
  while (!cached_results.empty())
    cached_results.pop();

  VLOG(3) << "Created RemoteIterator on table " << table->id() << ", shard "
             << shard << " @" << this;
  rpc::NetworkThread::Get()->Call(target_worker + 1, MessageTypes::ITERATOR,
      request_, &response_);
  for (size_t i = 1; i <= response_.row_count(); i++) {
    std::pair<string, string> row;
    row = make_pair(response_.key(i - 1), response_.value(i - 1));
    cached_results.push(row);
  }

  request_.set_id(response_.id());
}

bool RemoteIterator::done() {
  return response_.done() && cached_results.empty();
}

void RemoteIterator::next() {
  int target_worker = table_->worker_for_shard(shard_);
  if (!cached_results.empty()) cached_results.pop();

  if (cached_results.empty()) {
    if (response_.done()) {
      return;
    }
    rpc::NetworkThread::Get()->Call(target_worker + 1, MessageTypes::ITERATOR,
        request_, &response_);
    if (response_.row_count() < 1 && !response_.done())
    LOG(ERROR)<< "Call to server requesting " << request_.row_count()
    << " rows returned " << response_.row_count() << " rows.";
    for (size_t i = 1; i <= response_.row_count(); i++) {
      std::pair<string, string> row;
      row = make_pair(response_.key(i - 1), response_.value(i - 1));
      cached_results.push(row);
    }
  } else {
    VLOG(4) << "[PREFETCH] Using cached key for Next()";
  }
  ++index_;
}

const TableKey& RemoteIterator::key() {
  return cached_results.front().first;
}

const TableValue& RemoteIterator::value() {
  return cached_results.front().second;
}

int Table::shard_for_key(const TableKey& k) {
  return sharder->shard_for_key(k, this->num_shards());
}

const TableValue& Table::get_local(const TableKey& k) {
  int shard = this->shard_for_key(k);
  CHECK(is_local_shard(shard)) << " non-local for shard: " << shard;
  return (*shards_[shard])[k];
}

void Table::put(const TableKey& k, const TableValue& v) {
  int shard = this->shard_for_key(k);

  GRAB_LOCK;
  (*shards_[shard])[k] = v;

  if (!is_local_shard(shard)) {
    ++pending_writes_;
  }

  if (pending_writes_ > flush_frequency) {
    send_updates();
  }

  PERIODIC(0.1, {this->handle_put_requests();});
}

void Table::update(const TableKey& k, const TableValue& v) {
  int shard_id = this->shard_for_key(k);

  GRAB_LOCK;;

  Shard& s = (*shards_[shard_id]);
  Shard::iterator i = s.find(k);
  if (i == s.end()) {
    s[k] = v;
  } else {
    accum->accumulate(&i->second, v);
  }

  ++pending_writes_;
  if (pending_writes_ > flush_frequency) {
    send_updates();
  }

  PERIODIC(0.1, {this->handle_put_requests();});
}

const TableValue& Table::get(const TableKey& k) {
  int shard = this->shard_for_key(k);

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

  if (helper()->id() == -1) {
    LOG(INFO)<< "get" << is_local_shard(shard);
  }
  if (is_local_shard(shard)) {
    GRAB_LOCK;
    return (*shards_[shard])[k];
  }

  LOG(INFO)<< "Remote fetch...";
  TableValue* v = new TableValue;
  get_remote(shard, k, v);
  return *v;
}

bool Table::contains(const TableKey& k) {
  int shard = this->shard_for_key(k);

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
    Shard& s = (*shards_[shard]);
    return s.find(k) != s.end();
  }

  TableValue v;
  bool result = get_remote(shard, k, &v);
  return result;
}

void Table::remove(const TableKey& k) {
  LOG(FATAL)<< "Not implemented!";
}

Shard* Table::create_local(int shard_id) {
  return new Shard();
}

TableIterator* Table::get_iterator(int shard) {
  if (this->is_local_shard(shard)) {
    return new LocalIterator(*(shards_[shard]));
  } else {
    return new RemoteIterator(this, shard);
  }
}

void Table::update_partitions(const PartitionInfo& info) {
  shard_info_[info.shard()].CopyFrom(info);
}

Table::~Table() {
  for (auto p : shards_) {
    delete p;
  }
}

bool Table::is_local_shard(int shard) {
  if (!helper()) return false;
  return worker_for_shard(shard) == helper()->id();
}

int64_t Table::shard_size(int shard) {
  if (is_local_shard(shard)) {
    return shards_[shard]->size();
  } else {
    return shard_info_[shard].entries();
  }
}

bool Table::get_remote(int shard, const TableKey& k, TableValue* v) {
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

  req.set_key(k);
  req.set_table(id());
  req.set_shard(shard);

  if (!helper()) {
    LOG(FATAL)<< "get_remote() failed: helper() undefined.";
  }
  int peer = helper()->peer_for_shard(id(), shard);

  DCHECK_GE(peer, 0);
  DCHECK_LT(peer, rpc::NetworkThread::Get()->size() - 1);

  VLOG(2) << "Sending get request to: " << MP(peer, shard);
  rpc::NetworkThread::Get()->Call(peer + 1, MessageTypes::GET, req, &resp);

  if (resp.missing_key()) {
    return false;
  }

  *v = resp.kv_data(0).value();

  boost::recursive_mutex::scoped_lock sl(mutex());
  CacheEntry c = { Now(), *v };
  cache_[k] = c;
  return true;
}

void Table::clear() {
  ClearTable req;

  req.set_table(this->id());
  VLOG(2) << StringPrintf("Sending clear request (%d)", req.table());

  rpc::NetworkThread::Get()->SyncBroadcast(MessageTypes::CLEAR_TABLE, req);
}

void Table::start_checkpoint(const string& f, bool deltaOnly) {
  LOG(FATAL)<< "Not implemented.";
}

void Table::finish_checkpoint() {
  LOG(FATAL)<< "Not implemented.";
}

void Table::write_delta(const TableData& d) {
  LOG(FATAL)<< "Not implemented.";
}

void Table::restore(const string& f) {
  LOG(FATAL)<< "Not implemented.";
}

void Table::handle_put_requests() {
  helper()->handle_put_request();
}

int Table::send_updates() {
  int count = 0;

  TableData put;
  boost::recursive_mutex::scoped_lock sl(mutex());
  for (size_t i = 0; i < shards_.size(); ++i) {
    Shard* t = shards_[i];
    if (!is_local_shard(i) && (shard_info_[i].dirty() || !t->empty())) {
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
            MessageTypes::PUT_REQUEST, put);
      } while (!t->empty());

      t->clear();
    }
  }

  pending_writes_ = 0;
  return count;
}

int Table::pending_writes() {
  int64_t s = 0;
  for (size_t i = 0; i < shards_.size(); ++i) {
    Shard *t = shards_[i];
    if (!is_local_shard(i)) {
      s += t->size();
    }
  }

  return s;
}

}
