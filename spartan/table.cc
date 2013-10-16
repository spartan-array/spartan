#include "spartan/table.h"
#include "spartan/util/registry.h"
#include "spartan/util/timer.h"

using namespace spartan;

namespace spartan {

static pthread_key_t ctx_key_ = 0;
static rpc::Mutex ctx_lock_;

TableContext* TableContext::get_context() {
  rpc::ScopedLock l(&ctx_lock_);
  CHECK(ctx_key_ != 0);
  auto result = (TableContext*) pthread_getspecific(ctx_key_);
  CHECK(result != NULL);
  return result;
}

void TableContext::set_context(TableContext* ctx) {
  rpc::ScopedLock l(&ctx_lock_);
  if (ctx_key_ == 0) {
    pthread_key_create(&ctx_key_, NULL);
  }
  pthread_setspecific(ctx_key_, ctx);
}

int Table::flush() {
  int count = 0;
  TableData put;
  rpc::FutureGroup g;
  for (size_t i = 0; i < shards_.size(); ++i) {
    if (!is_local_shard(i)) {
      put.kv_data.clear();

      {
        GRAB_LOCK;
        Shard* t = (Shard*) shards_[i];
        for (auto j : *t) {
          put.kv_data.push_back( {j.first, j.second});
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
  // Wait for any updates that were sent asynchronously
  // (this occurs if we don't have a combiner).
  while (pending_updates_ > 0) {
    Sleep(0.0001);
  }
  return count;
}

bool Table::get_remote(int shard, const RefPtr& k, RefPtr* v) {
  Timer t;

  GetRequest req;
  TableData resp;
  req.key = k;
  req.table = id();
  req.shard = shard;
  if (!ctx()) {
    Log_fatal("get_remote() failed: helper() undefined.");
  }
  int peer = worker_for_shard(shard);

  CHECK_NE(peer, ctx()->id());

  //    Log_debug("Sending get request to: (%d, %d)", peer, shard);
  workers[peer]->get(req, &resp);

  Log_debug("Remote get %d -> %d finished in %.9f seconds.",
      peer, ctx()->id(), t.elapsed());
  if (resp.missing_key) {
    return false;
  }
  if (v != NULL) {
    *v = resp.kv_data[0].value;
  }
  return true;
}

bool Table::_get(int shard, const RefPtr& k, RefPtr* v) {
  if (shard == -1) {
    shard = this->shard_for_key(k);
  }
  while (tainted(shard)) {
    sched_yield();
  }

  if (is_local_shard(shard)) {
    GRAB_LOCK;
    Shard& s = (Shard&) (*shards_[shard]);
    typename Shard::iterator i = s.find(k);
    if (i == s.end()) {
      Log_fatal("WTF:: %s", repr(k).c_str());
      return false;
    }

    if (v != NULL) {
      if (selector != NULL) {
        *v = selector->select(k, i->second);
      } else {
        *v = i->second;
      }
    }

    return true;
  }
  return get_remote(shard, k, v);
}

void Table::update(int shard, const RefPtr& k, const RefPtr& v) {
  if (shard == -1) {
    shard = this->shard_for_key(k);
  }
  CHECK_LT(shard, num_shards());
  Shard& s = typed_shard(shard);
  typename Shard::iterator i = s.find(k);
  GRAB_LOCK;

  if (is_local_shard(shard)) {
    CHECK_NE(k.get(), NULL);
    if (i == s.end() || reducer == NULL) {
      s.insert(k, v);
    } else {
      reducer->accumulate(k, &i->second, v);
      CHECK_NE(i->second.get(), NULL);
    }
    return;
  }
  if (combiner != NULL) {
    if (i == s.end()) {
      s.insert(k, v);
    } else {
      combiner->accumulate(k, &i->second, v);
    }

    return;
  }
  ++pending_updates_;


  TableData put;
  put.table = this->id();
  put.shard = shard;
  put.kv_data.push_back( { k, v });
  auto callback = [=](rpc::Future *future) {
    GRAB_LOCK;
    --this->pending_updates_;
  };
  workers[worker_for_shard(shard)]->async_put(put, rpc::FutureAttr(callback));
}

RemoteIterator::RemoteIterator(Table *table, int shard, uint32_t fetch_num) :
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

bool RemoteIterator::done() {
  wait_for_fetch();
  return response_.done && index_ >= response_.results.size();
}

void RemoteIterator::wait_for_fetch() {
  if (pending_ != NULL) {
    pending_->wait();
    pending_->get_reply() >> response_;
    pending_ = NULL;
  }
}

void RemoteIterator::next() {
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

RefPtr RemoteIterator::key() {
  wait_for_fetch();
  return response_.results[index_].key;
}

RefPtr RemoteIterator::value() {
  wait_for_fetch();
  return response_.results[index_].value;
}

class Modulo: public Sharder {
  size_t shard_for_key(const RefPtr& k, int num_shards) const {
    return boost::hash_value(k) % num_shards;
  }
  DECLARE_REGISTRY_HELPER(Sharder, Modulo);
};
DEFINE_REGISTRY_HELPER(Sharder, Modulo);

}
