#include <signal.h>

#include <boost/bind.hpp>
#include <boost/tuple/tuple.hpp>
#include <boost/tuple/tuple_io.hpp>

#include "sparrow/kernel.h"
#include "sparrow/table.h"
#include "sparrow/util/common.h"
#include "sparrow/util/stats.h"
#include "sparrow/util/timer.h"
#include "sparrow/worker.h"

using boost::make_tuple;
using boost::unordered_map;
using boost::unordered_set;

namespace sparrow {

Worker::Worker(rpc::PollMgr* poller) {
  running_ = true;
  kernel_active_ = false;
  current_iterator_id_ = 0;
}

Worker::~Worker() {
  running_ = false;

  for (size_t i = 0; i < peers_.size(); ++i) {
    delete peers_[i];
  }
}

void Worker::run_kernel(const RunKernelReq& kreq) {
  int owner = tables_[kreq.table]->worker_for_shard(kreq.shard);
  if (owner != id_) {
    Log::fatal("Received a shard I can't work on! (Worker: %d, Shard: (%d, %d), Owner: %d)",
        id_, kreq.table, kreq.shard, owner);
  }

  Kernel* k = TypeRegistry<Kernel>::get_by_name(kreq.kernel);

  k->init(this, kreq.table, kreq.shard, kreq.args);

  k->run();
  flush();
}

void Worker::flush() {
  boost::recursive_try_mutex::scoped_lock sl(state_lock_);

  for (auto i : tables_) {
    i.second->flush();
  }
}

void Worker::get(const GetRequest& req, TableData* resp) {
  resp->source = id_;
  resp->table = req.table;
  resp->shard = -1;
  resp->done = true;

  {
    Table *t = tables_[req.table];
    if (!t->contains_str(req.key)) {
      resp->missing_key = true;
    } else {
      resp->missing_key = false;
      resp->kv_data.push_back( { req.key, t->get_str(req.key) });
    }
  }
}

void Worker::get_iterator(const IteratorReq& req, IteratorResp* resp) {
  int table = req.table;
  int shard = req.shard;

  Table * t = tables_[table];
  TableIterator* it = NULL;
  if (req.id == -1) {
    it = t->get_iterator(shard);
    uint32_t id = current_iterator_id_++;
    iterators_[id] = it;
    resp->id = id;
  } else {
    it = iterators_[req.id];
    resp->id = req.id;
    CHECK_NE(it, (void *)NULL);
    it->next();
  }

  for (size_t i = 0; i < req.count; i++) {
    resp->done = it->done();
    if (!it->done()) {
      resp->results.push_back( { it->key_str(), it->value_str() });
      resp->row_count = i;
      it->next();
    }
  }
}

void Worker::create_table(const CreateTableReq& req) {
  Table* t = TypeRegistry<Table>::get_by_id(req.table_type);
  t->init(req.id, req.num_shards);
  t->accum = TypeRegistry<Accumulator>::get_by_id(req.accum.type_id);
  t->accum->init(req.accum.opts);

  t->sharder = TypeRegistry<Sharder>::get_by_id(req.sharder.type_id);
  t->sharder->init(req.sharder.opts);
  if (req.selector.type_id != -1) {
//    LOG(INFO) << "Worker installing selector: " << req.selector_type();
    t->selector = TypeRegistry<Selector>::get_by_id(req.selector.type_id);
    t->selector->init(req.selector.opts);
  } else {
    t->selector = NULL;
  }

  t->set_ctx(this);
  tables_[req.id] = t;
}

void Worker::assign_shards(const ShardAssignmentReq& shard_req) {
//  Log::info("Shard assignment: " << shard_req.DebugString());
  for (auto a : shard_req.assign) {
    Table *t = tables_[a.table];
    t->shard_info(a.shard)->owner = a.new_worker;
  }
}

void Worker::shutdown() {
  running_ = false;
}

void Worker::delete_table(const DeleteTableReq& req) {
  tables_.erase(tables_.find(req.id));
}

void Worker::put(const TableData& req) {
  Table* t = tables_[req.table];
  for (auto p : req.kv_data) {
    t->update_str(p.key, p.value);
  }
}

void Worker::initialize(const WorkerInitReq& req) {

}

Worker* start_worker(const std::string& master_addr) {
  rpc::PollMgr* manager = new rpc::PollMgr;
  rpc::ThreadPool* threadpool = new rpc::ThreadPool;

  int port = rpc::find_open_port();
  RegisterReq req;
  req.addr.host = rpc::get_host_name();
  req.addr.port = port;

  rpc::Server* server = new rpc::Server(manager, threadpool);
  auto worker = new Worker(manager);
  server->reg(worker);
  server->start(
      StringPrintf("%s:%d", req.addr.host.c_str(), req.addr.port).c_str());

  MasterProxy* master = connect<MasterProxy>(manager, master_addr);
  master->register_worker(req);

  return worker;
}

} // end namespace
