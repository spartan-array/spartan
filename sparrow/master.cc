#include <set>
#include <algorithm>
#include <boost/tuple/tuple.hpp>
#include <boost/tuple/tuple_io.hpp>

#include "gflags/gflags.h"

#include "sparrow/table.h"
#include "sparrow/master.h"
#include "sparrow/util/registry.h"

using std::map;
using std::vector;
using std::set;
using namespace boost::tuples;

DEFINE_string(dead_workers, "",
    "For failure testing; comma delimited list of workers to pretend have died.");
DEFINE_bool(work_stealing, true,
    "Enable work stealing to load-balance tasks between machines.");
DEFINE_bool(checkpoint, false, "If true, enable checkpointing.");
DEFINE_bool(restore, false, "If true, enable restore.");

DECLARE_string(checkpoint_write_dir);
DECLARE_string(checkpoint_read_dir);
DECLARE_double(sleep_time);

namespace sparrow {

typedef std::pair<int, int> ShardId;

class TaskState {
public:
  TaskState() {

  }
  TaskState(ShardId id, int64_t size) :
      id(id), size(size), stolen(false) {
  }

  ShardId id;
  int size;
  bool stolen;
};

typedef map<ShardId, TaskState> TaskMap;
typedef std::set<ShardId> ShardSet;

static void try_remove(TaskMap& t, ShardId id) {
  if (t.find(id) != t.end()) {
    t.erase(t.find(id));
  }
}

class WorkerState: private boost::noncopyable {
public:
  WorkerState(int w_id) :
      id(w_id) {
    last_ping_time = Now();
    last_task_start = 0;
    total_runtime = 0;
    checkpointing = false;
    alive = true;
    status = 0;
  }

  TaskMap pending;
  TaskMap active;
  TaskMap finished;

  // Table shards this worker is responsible for serving.
  ShardSet shards;

  double last_ping_time;

  int status;
  int id;

  double last_task_start;
  double total_runtime;

  bool checkpointing;
  bool alive;

  bool is_assigned(ShardId id) {
    return pending.find(id) != pending.end();
  }

  void ping() {
    last_ping_time = Now();
  }

  double idle_time() {
    return Now() - last_ping_time;
  }

  bool serves(ShardId id) const {
    return shards.find(id) != shards.end();
  }

  void assign_shard(int table, int shard) {
    ShardId t(table, shard);
    shards.insert(t);
  }

  void assign_task(ShardId id) {
    TaskState state(id, 1);
    pending[id] = state;
  }

  void remove_task(ShardId id) {
    pending.erase(pending.find(id));
  }

  void clear_tasks() {
    CHECK(active.empty());
    pending.clear();
    active.clear();
    finished.clear();
  }

  void set_finished(const ShardId& id) {
    finished[id] = active[id];
    active.erase(active.find(id));
  }

  std::string str() {
    return StringPrintf("W(%d) p: %d; a: %d f: %d", id, pending.size(),
        active.size(), finished.size());
  }

  int num_assigned() const {
    return pending.size() + active.size() + finished.size();
  }

  int64_t total_size() const {
    int64_t out = 0;
    for (TaskMap::const_iterator i = pending.begin(); i != pending.end(); ++i) {
      out += 1 + i->second.size;
    }
    return out;
  }

  // Order pending tasks by our guess of how large they are
  bool get_next(const RunDescriptor& r, KernelRequest* msg) {
    if (pending.empty()) {
      return false;
    }

    TaskState state = pending.begin()->second;
    active[state.id] = state;
    pending.erase(pending.begin());

    msg->set_kernel(r.kernel);
    msg->set_table(r.table->id());
    msg->set_shard(state.id.second);
    for (auto i : r.args) {
      KV* kv = msg->add_args();
      kv->set_key(i.first);
      kv->set_value(i.second);
    }

    last_task_start = Now();

    return true;
  }
};

Master::Master() {
  kernel_epoch_ = 0;
  finished_ = dispatched_ = 0;

  network_ = rpc::NetworkThread::Get();
  shards_assigned_ = false;

  CHECK_GT(network_->size(), 1)<< "At least one master and one worker required!";

  for (int i = 0; i < num_workers(); ++i) {
    workers_.push_back(new WorkerState(i));
  }

  for (int i = 0; i < num_workers(); ++i) {
    RegisterWorkerRequest req;
    int src = 0;
    network_->Read(rpc::ANY_SOURCE, MessageTypes::REGISTER_WORKER, &req, &src);
    VLOG(1) << "Registered worker " << src - 1 << "; " << num_workers() - i
               << " remaining.";
  }

  LOG(INFO)<< "All workers registered; starting up.";

  vector<StringPiece> bits = StringPiece::split(FLAGS_dead_workers, ",");
//  LOG(INFO) << "dead workers: " << FLAGS_dead_workers;
  for (size_t i = 0; i < bits.size(); ++i) {
    LOG(INFO)<< make_pair(i, bits[i].AsString());
    workers_[i]->alive = false;
  }
}

Master::~Master() {
  VLOG(1) << "Total runtime: " << runtime_.elapsed();

  VLOG(1) << "Worker execution time:";
  if (VLOG_IS_ON(1)) {
    for (size_t i = 0; i < workers_.size(); ++i) {
      WorkerState& w = *workers_[i];
      if (i % 10 == 0) {
        fprintf(stderr, "\n%zu: ", i);
      }
      fprintf(stderr, "%.3f ", w.total_runtime);
    }
    fprintf(stderr, "\n");
  }

  VLOG(1) << "Kernel stats: ";
  for (MethodStatsMap::iterator i = method_stats_.begin();
      i != method_stats_.end(); ++i) {
    VLOG(1) << i->first << "--> " << i->second.ShortDebugString();
  }

  VLOG(1) << "Shutting down workers.";
  EmptyMessage msg;
  for (int i = 1; i < network_->size(); ++i) {
    network_->Send(i, MessageTypes::WORKER_SHUTDOWN, msg);
  }
}

void Master::start_checkpoint() {
}

void Master::finish_checkpoint() {
}

void Master::checkpoint() {
  start_checkpoint();
  finish_checkpoint();
}

WorkerState* Master::worker_for_shard(int table, int shard) {
  for (size_t i = 0; i < workers_.size(); ++i) {
    if (workers_[i]->serves(ShardId(table, shard))) {
      return workers_[i];
    }
  }

  return NULL;
}

WorkerState* Master::assign_shard(int table, int shard) {
  {
    WorkerState* ws = worker_for_shard(table, shard);
    if (ws != NULL) {
      return ws;
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
  p->set_owner(best->id);
  p->set_shard(shard);
  p->set_table(table);
  best->assign_shard(table, shard);

  return best;
}

void Master::send_table_assignments() {
  ShardAssignmentRequest req;

  for (size_t i = 0; i < workers_.size(); ++i) {
    WorkerState& w = *workers_[i];
    for (ShardSet::iterator j = w.shards.begin(); j != w.shards.end(); ++j) {
      ShardAssignment* s = req.add_assign();
      s->set_new_worker(i);
      s->set_table(j->first);
      s->set_shard(j->second);
    }
  }

  network_->SyncBroadcast(MessageTypes::SHARD_ASSIGNMENT, req);
  LOG(INFO)<< "Sent table assignments.";
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

  LOG(INFO)<< "Assigning workers for " << shards.size() << " shards.";
  for (auto i : shards) {
    int worker = r.table->shard_info(i)->owner();
    VLOG(1) << "Assigning shard: " << i << " to worker: " << worker;
    workers_[worker]->assign_task(ShardId(r.table->id(), i));
  }
}

int Master::dispatch_work(const RunDescriptor& r) {
  int num_dispatched = 0;
  KernelRequest w_req;
  for (size_t i = 0; i < workers_.size(); ++i) {
    WorkerState& w = *workers_[i];
    if (w.get_next(r, &w_req)) {
       VLOG(1) << "Dispatching: " << w.str() << " : " << w_req;
      num_dispatched++;
      network_->Send(w.id + 1, MessageTypes::RUN_KERNEL, w_req);
    }
  }
  return num_dispatched;
}

void Master::dump_stats() {
  string status;
  for (int k = 0; k < num_workers(); ++k) {
    status += StringPrintf("%d/%d ",
        workers_[k]->finished.size(),
        workers_[k]->num_assigned());
  }
  LOG(INFO)<< StringPrintf("Running %s (%d); %s; assigned: %d done: %d",
      current_run_.kernel.c_str(),
      current_run_.shards.size(),
      status.c_str(),
      dispatched_,
      finished_);

}

int Master::reap_one_task() {
  MethodStats &mstats = method_stats_[ToString(current_run_.kernel)];
  KernelDone done_msg;
  int w_id = 0;

  if (network_->TryRead(rpc::ANY_SOURCE, MessageTypes::KERNEL_DONE, &done_msg,
      &w_id)) {
    // (worker-id) == MPI id - 1.
    w_id -= 1;

    WorkerState& w = *workers_[w_id];

    ShardId task_id(done_msg.kernel().table(), done_msg.kernel().shard());
    w.set_finished(task_id);

    w.total_runtime += Now() - w.last_task_start;
    mstats.set_shard_time(mstats.shard_time() + Now() - w.last_task_start);
    mstats.set_shard_calls(mstats.shard_calls() + 1);
    w.ping();
    return w_id;
  } else {
    Sleep(FLAGS_sleep_time);
    return -1;
  }

}

void Master::run(RunDescriptor r) {
  flush_network();

  CHECK_EQ(current_run_.shards.size(), finished_)<<
  " Cannot start kernel before previous one is finished ";
  finished_ = dispatched_ = 0;

  Kernel::ScopedPtr k(TypeRegistry<Kernel>::get_by_name(r.kernel));
  CHECK_NE(k.get(), (void*)NULL)<< "Invalid kernel class " << r.kernel;

  VLOG(1) << "Running: " << r.kernel << " on table " << r.table->id();

  vector<int> shards = r.shards;

  MethodStats &mstats = method_stats_[ToString(r.kernel)];
  mstats.set_calls(mstats.calls() + 1);

  current_run_ = r;
  current_run_start_ = Now();

  kernel_epoch_++;

  VLOG(1) << "Current run: " << shards.size() << " shards";
  assign_tasks(current_run_, shards);

  dispatched_ += dispatch_work(current_run_);

  barrier();
}

void Master::barrier() {
  MethodStats &mstats = method_stats_[ToString(current_run_.kernel)];

  while (finished_ < current_run_.shards.size()) {
    PERIODIC(10, { DumpProfile(); dump_stats(); });

    dispatched_ += dispatch_work(current_run_);

    if (reap_one_task() >= 0) {
      finished_++;
    }
  }

  // Force workers to flush outputs.
  flush_network();

  // Force workers to apply flushed updates.
  flush_network();

  mstats.set_total_time(mstats.total_time() + Now() - current_run_start_);
  LOG(INFO)<< "Kernel '" << current_run_.kernel << "' finished in " << Now() - current_run_start_;
}

void Master::flush_network() {
  // Flush any pending table updates
  for (auto i : tables_) {
    i.second->flush();
  }

  EmptyMessage empty;
  network_->SyncBroadcast(MessageTypes::WORKER_FLUSH, empty);
}

} // namespace sparrow
