#include <set>
#include <algorithm>
#include <boost/tuple/tuple.hpp>
#include <boost/tuple/tuple_io.hpp>

#include "spartan/table.h"
#include "spartan/master.h"
#include "spartan/util/registry.h"

using std::map;
using std::vector;
using std::set;
using namespace boost::tuples;

namespace spartan {

Master::Master(int num_workers) {
  num_workers_ = num_workers;
  current_run_start_ = 0;
  client_poller_ = new rpc::PollMgr;
  initialized_ = false;
  table_id_counter_ = 0;

  TableContext::set_context(this);
}

void Master::wait_for_workers() {
  if (initialized_) {
    return;
  }

  while (workers_.size() < num_workers_) {
    PERIODIC(5,
        Log_info("Waiting for workers... %d/%d", workers_.size(), num_workers_));
    Sleep(0.1);
  }
  Log_info("All workers registered; starting up.");

  WorkerInitReq req;
  for (auto w : workers_) {
    req.workers[w->id] = w->addr;
  }

  for (auto w : workers_) {
    req.id = w->id;
    w->proxy->initialize(req);
  }

  initialized_ = true;
}

Master::~Master() {
  shutdown();
}

void Master::shutdown() {
  for (auto t : tables_) {
    destroy_table(t.second);
  }

  CHECK(tables_.empty());

  for (auto w : workers_) {
    w->proxy->shutdown();
  }

  workers_.clear();
  tables_.clear();
}

void Master::register_worker(const RegisterReq& req) {
  rpc::ScopedLock sl(&lock_);
  int worker_id = workers_.size();
  WorkerState* w = new WorkerState(worker_id, req.addr);

  w->proxy = connect<WorkerProxy>(client_poller_,
      StringPrintf("%s:%d", req.addr.host.c_str(), req.addr.port));
  workers_.push_back(w);
}

WorkerState* Master::assign_shard(int table, int shard) {
  {
    int owner = tables_[table]->shard_info(shard)->owner;
    if (owner != -1) {
      return workers_[owner];
    }
  }

  WorkerState* best = NULL;
  for (size_t i = 0; i < workers_.size(); ++i) {
    WorkerState& w = *workers_[i];
    if (!w.alive) {
      continue;
    }

    if (w.serves_shard(shard)) {
      best = &w;
      break;
    }

    if (best == NULL || w.shards.size() < best->shards.size()) {
      best = &w;
    }
  }

  CHECK(best != NULL);
  CHECK(best->alive);

  Log_debug("Assigned shard (%d, %d) to worker %d", table, shard, best->id);

  // Update local partition information, for performing put/fetches
  // on the master.
  PartitionInfo* p = tables_[table]->shard_info(shard);
  p->owner = best->id;
  p->shard = shard;
  p->table = table;
  best->shards.insert(ShardId(table, shard));

  return best;
}

void Master::send_table_assignments() {
  ShardAssignmentReq req;

  for (auto i : tables_) {
    auto t = i.second;
    for (int j = 0; j < t->num_shards(); ++j) {
      req.assign.push_back( { t->id(), j, t->shard_info(j)->owner } );
    }
  }

  for (auto w : workers_) {
    w->proxy->assign_shards(req);
  }
  Log_debug("Sent table assignments.");
}

void Master::assign_shards(Table* t) {
  for (int j = 0; j < t->num_shards(); ++j) {
    assign_shard(t->id(), j);
  }

  send_table_assignments();
}

void Master::assign_tasks(const RunDescriptor& r, vector<int> shards) {
  for (auto w : workers_) {
    w->clear_tasks();
  }

  Log_debug("Assigning workers for %d shards.", shards.size());
  for (auto i : shards) {
    int worker = r.table->shard_info(i)->owner;
    workers_[worker]->assign_task(ShardId(r.table->id(), i));
  }
}

int Master::dispatch_work(const RunDescriptor& r) {
  int num_dispatched = 0;
  for (size_t i = 0; i < workers_.size(); ++i) {
    WorkerState* w = workers_[i];
    RunKernelReq w_req;
    if (!w->get_next(r, &w_req)) {
      continue;
    }

    auto callback = [=](rpc::Future *future) {
      Log_debug("MASTER: Kernel %d:%d finished", w_req.table, w_req.shard);
      w->set_finished(ShardId(w_req.table, w_req.shard));
    };

    rpc::Future *f = w->proxy->async_run_kernel(w_req, rpc::FutureAttr(callback));
    running_kernels_[w_req.shard] = f;
//    assert(w->proxy->run_kernel(w_req)== 0);
    Log_debug("MASTER: Kernel %d:%d dispatched as request %p", w_req.table, w_req.shard, f);
    num_dispatched++;
  }
  return num_dispatched;
}

int Master::num_pending(const RunDescriptor& r) {
  int t = 0;
  for (auto w : workers_) {
    t += w->num_pending();
  }
  return t;
}

void Master::destroy_table(int table_id) {
  if (tables_.find(table_id) == tables_.end()) {
    return;
  }

  CHECK(tables_.find(table_id) != tables_.end());
  Log_debug("Destroying table %d", table_id);
  for (auto w : workers_) {
    w->proxy->destroy_table(table_id);
    for (auto s = w->shards.begin(); s != w->shards.end(); ) {
      if (s->first == table_id) {
        s = w->shards.erase(s);
      } else {
        ++s;
      }
    }
  }

  Table* t = tables_[table_id];
  delete t;
  tables_.erase(tables_.find(table_id));
}

void Master::run(RunDescriptor r) {
  wait_for_workers();
  flush();

  Kernel::ScopedPtr k(TypeRegistry<Kernel>::get_by_name(r.kernel));
  CHECK_NE(k.get(), (void*)NULL);

  Log_info("Running: %s on %d", r.kernel.c_str(), r.table->id());

  vector<int> shards = r.shards;

  current_run_ = r;
  current_run_start_ = Now();

  assign_tasks(current_run_, shards);

  dispatch_work(current_run_);
  while (num_pending(r) > 0) {
    dispatch_work(current_run_);
//    Log_info("Dispatch loop: %d", running_kernels_.size());
    Sleep(0);
  }

  int count = 0;
  for (auto f : running_kernels_) {
    Log_debug("Waiting for kernel %d/%d to finish...", count, running_kernels_.size());
    f.second->wait();
    f.second->release();
    ++count;
  }

  running_kernels_.clear();

  // Force workers to flush outputs.
  flush();

  // Force workers to apply flushed updates.
  flush();

  Log_info("Kernel %s finished in %f", current_run_.kernel.c_str(),
      Now() - current_run_start_);
}

void Master::flush() {
  // Flush any pending table updates
  for (auto i : tables_) {
    i.second->flush();
  }

  rpc::FutureGroup g;
  for (auto w : workers_) {
    g.add(w->proxy->async_flush());
  }

  //Log_info("Flushed...");
}

Master* start_master(int port, int num_workers) {
  auto poller = new rpc::PollMgr;
  auto tpool = new rpc::ThreadPool(8);
  auto server = new rpc::Server(poller, tpool);

  auto master = new Master(num_workers);
  server->reg(master);
  auto hostname = rpc::get_host_name();
  server->start(StringPrintf("%s:%d", hostname.c_str(), port).c_str());
  return master;
}

} // namespace spartan
