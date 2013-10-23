#include <set>
#include <algorithm>
#include <boost/tuple/tuple.hpp>
#include <boost/tuple/tuple_io.hpp>

#include "spartan/table.h"
#include "spartan/master.h"
#include "spartan/util/registry.h"
#include "spartan/py-support.h"

using std::map;
using std::vector;
using std::set;
using namespace boost::tuples;

namespace spartan {

class WorkerState: private boost::noncopyable {
public:
  TaskMap pending;
  TaskMap active;
  TaskMap finished;

  map<ShardId, string> errors;

  ShardSet shards;

  int status;
  int id;

  double last_ping_time;
  double total_runtime;

  bool alive;

  WorkerProxy *proxy;
  HostPort addr;

  mutable rpc::Mutex lock;

  WorkerState(int w_id, HostPort addr) :
      id(w_id) {
    last_ping_time = Now();
    total_runtime = 0;
    alive = true;
    status = 0;
    proxy = NULL;
    this->addr = addr;
  }

  bool is_assigned(ShardId id) {
    rpc::ScopedLock sl(&lock);

    return pending.find(id) != pending.end();
  }

  bool serves_shard(int shard) {
    for (auto sid : shards) {
      if (sid.shard == shard) {
        return true;
      }
    }
    return false;
  }

  void ping() {
    last_ping_time = Now();
  }

  double idle_time() {
    return Now() - last_ping_time;
  }

  void assign_task(ShardId id, ArgMap args) {
    rpc::ScopedLock sl(&lock);

    TaskState state(id, 1, args);
    pending.insert(make_pair(id, state));
  }

  void remove_task(ShardId id) {
    rpc::ScopedLock sl(&lock);

    pending.erase(pending.find(id));
  }



  void clear_tasks() {
    rpc::ScopedLock sl(&lock);

    CHECK(active.empty());
    pending.clear();
    active.clear();
    finished.clear();
    errors.clear();
  }

  void set_finished(const ShardId& id, double kernel_time, string error) {
    rpc::ScopedLock sl(&lock);

    if (!error.empty()) {
      Log_debug("Error detected for task %d.%d (%s)", id.table, id.shard, error.c_str());
      errors[id] = error;
    }

    auto it = active.find(id);
    finished.insert(*it);
    active.erase(it);

    total_runtime += kernel_time;
  }

  std::string str() {
    rpc::ScopedLock sl(&lock);

    return StringPrintf("W(%d) p: %d; a: %d f: %d", id, pending.size(),
        active.size(), finished.size());
  }

  int num_assigned() const {
    rpc::ScopedLock sl(&lock);
    return pending.size() + active.size() + finished.size();
  }

  int num_pending() const {
    return pending.size();
  }

  bool can_schedule() const {
    return !pending.empty() && active.empty();
  }

  // Order pending tasks by our guess of how large they are
  void get_next(RunKernelReq* msg) {
    rpc::ScopedLock sl(&lock);

    CHECK_EQ(can_schedule(), true);

    TaskState state = pending.begin()->second;
    active.insert(make_pair(state.id, state));
    pending.erase(pending.begin());

    msg->table = state.id.table;
    msg->shard = state.id.shard;

    for (auto i : state.args) {
      msg->task_args.insert(i);
    }
  }
};

WorkerProxy* worker_proxy(WorkerState* w) {
  return w->proxy;
}

int worker_id(WorkerState* w) {
  return w->id;
}

Master::Master(int num_workers) {
  num_workers_ = num_workers;
  client_poller_ = new rpc::PollMgr;
  initialized_ = false;
  table_id_counter_ = 0;
  server_ = NULL;
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

  rpc::FutureGroup g;
  for (auto w : workers_) {
    req.id = w->id;
    g.add(w->proxy->async_initialize(req));
  }
  g.wait_all();

  initialized_ = true;
}

Master::~Master() {
  shutdown();
}

void Master::shutdown() {
  rpc::FutureGroup g;
  for (auto w : workers_) {
    Log_debug("Shutting down %s:%d", w->addr.host.c_str(), w->addr.port);
    Log_info("Worker %d (%s:%d) %f",
        w->id, w->addr.host.c_str(), w->addr.port, w->total_runtime);
    g.add(w->proxy->async_shutdown());
  }
  g.wait_all();

  workers_.clear();

  delete server_;
  server_ = NULL;
}

void Master::register_worker(const RegisterReq& req) {
  rpc::ScopedLock sl(&lock_);
  int worker_id = workers_.size();
  WorkerState* w = new WorkerState(worker_id, req.addr);

  w->proxy = connect<WorkerProxy>(client_poller_,
      StringPrintf("%s:%d", req.addr.host.c_str(), req.addr.port));
  workers_.push_back(w);
}

Table* Master::create_table(
    Sharder* sharder,
    Accumulator* combiner,
    Accumulator* reducer,
    Selector* selector) {
  Timer timer;
  wait_for_workers();
  Table* t = new Table;
  // Crash here if we can't find the sharder/accumulators.
  delete TypeRegistry<Sharder>::get_by_id(sharder->type_id());
  CreateTableReq req;
  int table_id = table_id_counter_++;
  Log_debug("Creating table %d", table_id);
  req.id = table_id;

  req.num_shards = workers_.size();

  if (combiner != NULL) {
    delete TypeRegistry<Accumulator>::get_by_id(combiner->type_id());
    req.combiner.type_id = combiner->type_id();
    req.combiner.opts = combiner->opts();
  } else {
    req.combiner.type_id = -1;
  }
  if (reducer != NULL) {
    delete TypeRegistry<Accumulator>::get_by_id(reducer->type_id());
    req.reducer.type_id = reducer->type_id();
    req.reducer.opts = reducer->opts();
  } else {
    req.reducer.type_id = -1;
  }
  if (sharder != NULL) {
    req.sharder.type_id = sharder->type_id();
    req.sharder.opts = sharder->opts();
  } else {
    req.sharder.type_id = -1;
  }
  if (selector != NULL) {
    req.selector.type_id = selector->type_id();
    req.selector.opts = selector->opts();
  } else {
    req.selector.type_id = -1;
  }
  t->init(table_id, req.num_shards);
  t->sharder = sharder;
  t->combiner = combiner;
  t->reducer = reducer;
  t->selector = selector;
  t->workers.resize(workers_.size());
  for (auto w : workers_) {
    t->workers[worker_id(w)] = worker_proxy(w);
  }
  t->set_ctx(this);
  tables_[t->id()] = t;
  rpc::FutureGroup futures;
  for (auto w : workers_) {
    futures.add(worker_proxy(w)->async_create_table(req));
  }
  futures.wait_all();
  assign_shards(t);
  Log_debug("Table created in %f seconds", timer.elapsed());
  return t;
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
      req.assign.push_back( { t->id(), j, t->shard_info(j)->owner });
    }
  }

  rpc::FutureGroup g;
  for (auto w : workers_) {
    g.add(w->proxy->async_assign_shards(req));
  }
  g.wait_all();
  Log_debug("Sent table assignments.");
}

void Master::assign_shards(Table* t) {
  for (int j = 0; j < t->num_shards(); ++j) {
    assign_shard(t->id(), j);
  }

  send_table_assignments();
}

int Master::dispatch_work(RunState &r) {
  int num_dispatched = 0;

  for (size_t i = 0; i < workers_.size(); ++i) {
    WorkerState* w = workers_[i];
    if (!w->can_schedule()) {
      continue;
    }

    // setup kernel level arguments;
    // task arguments are filled in by get_next
    RunKernelReq w_req;
    w_req.kernel_args = r.kernel_args;
    w_req.kernel = r.kernel->type_id();
    w->get_next(&w_req);

    double start_time = Now();
    int table = w_req.table;
    int shard = w_req.shard;

    auto callback = [=](rpc::Future *future) {
      RunKernelResp resp;
      future->get_reply() >> resp;

      Log_debug("%d(%s:%d) %f",
          w->id, w->addr.host.c_str(), w->addr.port, resp.elapsed);
      w->set_finished(ShardId(table, shard),
                      Now() - start_time,
                      resp.error);
    };

    rpc::Future *f = w->proxy->async_run_kernel(w_req,
        rpc::FutureAttr(callback));
    running_kernels_[w_req.shard] = f;
//    assert(w->proxy->run_kernel(w_req)== 0);
    Log_debug("MASTER: Kernel %d:%d dispatched as request %p",
        w_req.table, w_req.shard, f);
    num_dispatched++;
  }
  return num_dispatched;
}

int Master::num_pending() {
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
  rpc::FutureGroup g;
  for (auto w : workers_) {
    g.add(w->proxy->async_destroy_table(table_id));

    for (auto s = w->shards.begin(); s != w->shards.end();) {
      if (s->table == table_id) {
        s = w->shards.erase(s);
      } else {
        ++s;
      }
    }
  }

  g.wait_all();

  Table* t = tables_[table_id];
  delete t;
  tables_.erase(tables_.find(table_id));
}

void Master::map_worklist(WorkList worklist, Kernel* k) {
  Log_fatal("Not implemented...");
  wait_for_workers();
  flush();

  for (auto w : workers_) {
    w->clear_tasks();
  }

  for (auto i : worklist) {
    auto t = tables_[i.locality.table];
    int worker = t->shard_info(i.locality.shard)->owner;
    workers_[worker]->assign_task(i.locality, i.args);
  }

  //wait_for_completion();
}

void Master::map_shards(Table* table, Kernel* k, ArgMap kernel_args) {
  RunState st;
  st.kernel = k;
  st.kernel_args = kernel_args;

  wait_for_workers();
  flush();

  int kernel_id = k->type_id();

  Kernel::ScopedPtr test_k(TypeRegistry<Kernel>::get_by_id(kernel_id));
  CHECK_NE(test_k.get(), (void*)NULL);

  Log_debug("Running: kernel %d against table %d", kernel_id, table->id());

  auto shards = range(0, table->num_shards());
  for (auto w : workers_) {
    w->clear_tasks();
  }

  Log_debug("Assigning workers for %d shards.", shards.size());
  for (auto i : shards) {
    int worker = table->shard_info(i)->owner;
    ArgMap task_args;
    task_args["shard"] = StringPrintf("%d", i);
    task_args["table"] = StringPrintf("%d", table->id());
    workers_[worker]->assign_task(ShardId(table->id(), i), task_args);
  }

  wait_for_completion(st);
}

void Master::wait_for_completion(RunState& r) {
  SleepBackoff sleeper(0.001);
  dispatch_work(r);

  while (num_pending() > 0) {
    if (dispatch_work(r) > 0) {
      sleeper.reset();
    } else {
      sleeper.sleep();
    }
  }

  int count = 0;
  for (auto f : running_kernels_) {
    Log_debug("Waiting for kernel %d/%d to finish...",
        count, running_kernels_.size());
    f.second->wait();
    f.second->release();
    ++count;
  }

  running_kernels_.clear();

  // Force workers to flush outputs.
  flush();
  for (size_t i = 0; i < workers_.size(); ++i) {
    WorkerState* w = workers_[i];
    if (!w->errors.empty()) {
      throw new PyException(w->errors.begin()->second);
    }
  }
  Log_debug("Kernel finished in %f", r.elapsed());
}

void Master::flush() {
  Timer t;
  // Flush any pending table updates
  for (auto i : tables_) {
    i.second->flush();
  }

  rpc::FutureGroup g;
  for (auto w : workers_) {
    g.add(w->proxy->async_flush());
  }

  g.wait_all();

  // Log_info("Flush took %f seconds.", t.elapsed());
}

Master* start_master(int port, int num_workers) {
  auto poller = new rpc::PollMgr;
  auto tpool = new rpc::ThreadPool(4);
  auto server = new rpc::Server(poller, tpool);

  auto master = new Master(num_workers);
  server->reg(master);
  auto hostname = rpc::get_host_name();
  server->start(StringPrintf("%s:%d", hostname.c_str(), port).c_str());

  master->set_server(server);
  return master;
}

} // namespace spartan
