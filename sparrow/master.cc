#include <set>
#include <algorithm>
#include <boost/tuple/tuple.hpp>
#include <boost/tuple/tuple_io.hpp>

#include "sparrow/table.h"
#include "sparrow/master.h"
#include "sparrow/util/registry.h"

using std::map;
using std::vector;
using std::set;
using namespace boost::tuples;

namespace sparrow {

Master::Master(rpc::PollMgr* poller, int num_workers) {
  num_workers_ = num_workers;
  current_run_start_ = 0;
  poller_ = poller;
}

void Master::wait_for_workers() {
  while (workers_.size() < num_workers_) {
    Sleep(0.01);
  }
  Log::info("All workers registered; starting up.");

  WorkerInitReq req;
  for (auto w : workers_) {
    req.workers[w->id] = w->addr;
  }

  for (auto w : workers_) {
    req.id = w->id;
    w->proxy->initialize(req);
  }
}

Master::~Master() {
  for (auto w : workers_) {
    w->proxy->shutdown();
  }
}

void Master::register_worker(const RegisterReq& req) {
  WorkerState* w = new WorkerState(workers_.size(), req.addr);
  w->proxy = connect<WorkerProxy>(poller_,
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
    if (w.alive && (best == NULL || w.shards.size() < best->shards.size())) {
      best = workers_[i];
    }
  }

  CHECK(best != NULL);
  CHECK(best->alive);

  // Update local partition information, for performing put/fetches
  // on the master.
  PartitionInfo* p = tables_[table]->shard_info(shard);
  p->owner = best->id;
  p->shard = shard;
  p->table = table;
  best->assign_shard(table, shard);

  return best;
}

void Master::send_table_assignments() {
  ShardAssignmentReq req;

  for (int i = 0; i < workers_.size(); ++i) {
    WorkerState& w = *workers_[i];
    for (ShardSet::iterator j = w.shards.begin(); j != w.shards.end(); ++j) {
      req.assign.push_back( { j->first, j->second, -1, i });
    }
  }

  for (auto w : workers_) {
    w->proxy->assign_shards(req);
  }
  Log::info("Sent table assignments.");
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

  Log::info("Assigning workers for %d shards.", shards.size());
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
    if (w->get_next(r, &w_req)) {
      Log::info("Dispatching: %s", w->str().c_str());
      num_dispatched++;

      auto callback = [=](rpc::Future *future) {
        Log::info("Finished.");
        w->set_finished(ShardId(w_req.table, w_req.shard));
      };
      running_kernels_.insert(
          w->proxy->async_run_kernel(w_req, rpc::FutureAttr(callback)));
    }
  }
  return num_dispatched;
}

void Master::run(RunDescriptor r) {
  flush();

  Kernel::ScopedPtr k(TypeRegistry<Kernel>::get_by_name(r.kernel));
  CHECK_NE(k.get(), (void*)NULL);

  Log::info("Running: %s on %d", r.kernel.c_str(), r.table->id());

  vector<int> shards = r.shards;

  current_run_ = r;
  current_run_start_ = Now();

  assign_tasks(current_run_, shards);

  dispatch_work(current_run_);
  while (!running_kernels_.empty()) {
    for (auto f : running_kernels_) {
      if (f->ready()) {
        f->release();
        running_kernels_.erase(f);
      }
    }

    dispatch_work(current_run_);
    Log::info("Dispatch loop: %d", running_kernels_.size());
    Sleep(0.1);
  }

  // Force workers to flush outputs.
  flush();

  // Force workers to apply flushed updates.
  flush();

  Log::info("Kernel %s finished in %f", current_run_.kernel.c_str(),
      Now() - current_run_start_);
}

void Master::flush() {
  // Flush any pending table updates
  for (auto i : tables_) {
    i.second->flush();
  }

  for (auto w : workers_) {
    w->proxy->flush();
  }
}

Master* start_master(int port, int num_workers) {
  auto poller = new rpc::PollMgr;
  auto tpool = new rpc::ThreadPool;
  auto server = new rpc::Server(poller, tpool);

  auto master = new Master(poller, num_workers);
  server->reg(master);
  auto hostname = rpc::get_host_name();
  server->start(StringPrintf("%s:%d", hostname.c_str(), port).c_str());
  return master;
}

} // namespace sparrow
