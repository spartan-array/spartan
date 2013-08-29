#include <signal.h>

#include <memory>
#include <boost/bind.hpp>
#include <boost/tuple/tuple.hpp>
#include <boost/tuple/tuple_io.hpp>

#include "spartan/kernel.h"
#include "spartan/table.h"
#include "spartan/util/common.h"
#include "spartan/util/timer.h"
#include "spartan/worker.h"

using boost::make_tuple;
using boost::unordered_map;
using boost::unordered_set;

namespace spartan {

Worker::Worker(rpc::PollMgr* poller) {
  running_ = true;
  current_iterator_id_ = 0;
  poller_ = poller;
  id_ = -1;
}

Worker::~Worker() {
  running_ = false;

  for (size_t i = 0; i < peers_.size(); ++i) {
    delete peers_[i];
  }

  for (auto t : tables_) {
    delete t.second;
  }
}

void Worker::run_kernel(const RunKernelReq& kreq) {
  TableContext::set_context(this);

  CHECK(id_ != -1);
  Log_debug("WORKER %d: Running kernel: %d:%d on %d items",
      id_, kreq.table, kreq.shard, tables_[kreq.table]->shard(kreq.shard)->size());
  int owner = -1;
  owner = tables_[kreq.table]->worker_for_shard(kreq.shard);

  if (owner != id_) {
    Log_fatal(
        "Received a shard I can't work on! (Worker: %d, Shard: (%d, %d), Owner: %d)",
        id_, kreq.table, kreq.shard, owner);
  }

  std::unique_ptr<Kernel> k(TypeRegistry<Kernel>::get_by_name(kreq.kernel));
  k->init(this, kreq.table, kreq.shard, kreq.args);
  k->run();

  Log_debug("WORKER: Finished kernel: %d:%d", kreq.table, kreq.shard);
}

void Worker::flush() {
  Log_debug("WORKER %d: Flushing tables...", id_);
  for (auto i : tables_) {
    i.second->flush();
  }
  Log_debug("WORKER %d: done.", id_);
}

void Worker::get(const GetRequest& req, TableData* resp) {
  Log_debug("WORKER %d: handling get.", id_);
  resp->source = id_;
  resp->table = req.table;
  resp->shard = -1;
  resp->done = true;

  Table *t = tables_[req.table];
  if (!t->contains_str(req.key)) {
    resp->missing_key = true;
  } else {
    resp->missing_key = false;
    resp->kv_data.push_back( { req.key, t->get_str(req.key) });
  }
  Log_debug("WORKER %d: done handling get.", id_);
}

void Worker::get_iterator(const IteratorReq& req, IteratorResp* resp) {
  int table = req.table;
  int shard = req.shard;

  Table * t = tables_[table];
  TableIterator* it = NULL;

  {
    rpc::ScopedLock sl(lock_);
    if (req.id == -1) {
      it = t->get_iterator(shard);
      uint32_t id = current_iterator_id_++;
      iterators_[id] = it;
      resp->id = id;
    } else {
      it = iterators_[req.id];
      resp->id = req.id;
      CHECK_NE(it, (void *)NULL);
    }
  }

  for (size_t i = 0; i < req.count; i++) {
    if (!it->done()) {
      resp->results.push_back( { it->key_str(), it->value_str() });
      resp->row_count = i;
      it->next();
    }
  }

  resp->done = it->done();
}

void Worker::create_table(const CreateTableReq& req) {
  CHECK(id_ != -1);
  rpc::ScopedLock sl(lock_);
  Log_debug("Creating table: %d", req.id);
  Table* t = TypeRegistry<Table>::get_by_id(req.table_type);
  t->init(req.id, req.num_shards);
  t->accum = TypeRegistry<Accumulator>::get_by_id(req.accum.type_id);
  t->accum->init(req.accum.opts);

  t->sharder = TypeRegistry<Sharder>::get_by_id(req.sharder.type_id);
  t->sharder->init(req.sharder.opts);
  if (req.selector.type_id != -1) {
    t->selector = TypeRegistry<Selector>::get_by_id(req.selector.type_id);
    t->selector->init(req.selector.opts);
  } else {
    t->selector = NULL;
  }

  t->workers = peers_;

  t->set_ctx(this);
  tables_[req.id] = t;
}

void Worker::assign_shards(const ShardAssignmentReq& shard_req) {
//  Log_info("Shard assignment: " << shard_req.DebugString());
  rpc::ScopedLock sl(lock_);
  for (auto a : shard_req.assign) {
    CHECK(tables_.find(a.table) != tables_.end());
    Table *t = tables_[a.table];
    t->shard_info(a.shard)->owner = a.worker;
  }
}

void Worker::shutdown() {
  for (auto t : tables_) {
    delete t.second;
  }
  tables_.clear();
  running_ = false;
}

void Worker::destroy_table(const rpc::i32& id) {
  rpc::ScopedLock sl(lock_);
//  Log_info("Destroying table %d", id);
  delete tables_[id];
  tables_.erase(tables_.find(id));
}

void Worker::put(const TableData& req) {
  Log_debug("WORKER: %d -- got put", id_);
  Table* t;
  {
    rpc::ScopedLock sl(lock_);
    t = tables_[req.table];
  }
  for (auto p : req.kv_data) {
    t->update_str(p.key, p.value);
  }
  Log_debug("WORKER: %d -- finished put", id_);
}

void Worker::initialize(const WorkerInitReq& req) {
  id_ = req.id;

  Log_info("Initializing worker %d, with connections to %d peers.",
      id_, req.workers.size());

  peers_.resize(req.workers.size());
  for (auto w : req.workers) {
    if (w.first != id_) {
      peers_[w.first] = connect<WorkerProxy>(poller_,
          StringPrintf("%s:%d", w.second.host.c_str(), w.second.port));
    }
  }
}

Worker* start_worker(const std::string& master_addr, int port) {
  rpc::PollMgr* manager = new rpc::PollMgr;
  rpc::ThreadPool* threadpool = new rpc::ThreadPool(8);

  if (port == -1) {
    port = rpc::find_open_port();
  }

  RegisterReq req;
  req.addr.host = rpc::get_host_name();
  req.addr.port = port;

  Log_info("Starting worker %d", port);
  rpc::Server* server = new rpc::Server(manager, threadpool);
  auto worker = new Worker(manager);
  server->reg(worker);
  server->start(
      StringPrintf("%s:%d", req.addr.host.c_str(), req.addr.port).c_str());

  MasterProxy* master = connect<MasterProxy>(manager, master_addr);
  Log_info("Registering %d", port);
  master->register_worker(req);
  Log_info("Done.", port);
  return worker;
}

} // end namespace
