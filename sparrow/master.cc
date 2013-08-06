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

static std::set<int> dead_workers;

struct Taskid {
  int table;
  int shard;

  Taskid(int t, int s) :
      table(t), shard(s) {
  }

  bool operator<(const Taskid& b) const {
    return table < b.table || (table == b.table && shard < b.shard);
  }
};

class TaskState: private boost::noncopyable {
public:
  enum Status {
    PENDING = 0, ACTIVE = 1, FINISHED = 2
  };

  TaskState(Taskid id, int64_t size) :
      id(id), status(PENDING), size(size), stolen(false) {
  }

  static bool IdCompare(TaskState *a, TaskState *b) {
    return a->id < b->id;
  }

  static bool WeightCompare(TaskState *a, TaskState *b) {
    if (a->stolen && !b->stolen) {
      return true;
    }
    return a->size < b->size;
  }

  Taskid id;
  int status;
  int size;
  bool stolen;
};

typedef map<Taskid, TaskState*> TaskMap;
typedef std::set<Taskid> ShardSet;

class WorkerState: private boost::noncopyable {
public:
  WorkerState(int w_id) :
      id(w_id) {
    last_ping_time = Now();
    last_task_start = 0;
    total_runtime = 0;
    checkpointing = false;
  }

  TaskMap work;

  // Table shards this worker is responsible for serving.
  ShardSet shards;

  double last_ping_time;

  int status;
  int id;

  double last_task_start;
  double total_runtime;

  bool checkpointing;

  // Order by number of pending tasks and last update time.
  static bool PendingCompare(WorkerState *a, WorkerState* b) {
//    return (a->pending_size() < b->pending_size());
    return a->num_pending() < b->num_pending();
  }

  bool alive() const {
    return dead_workers.find(id) == dead_workers.end();
  }

  bool is_assigned(Taskid id) {
    return work.find(id) != work.end();
  }

  void ping() {
    last_ping_time = Now();
  }

  double idle_time() {
    // Wait a little while before stealing work; should really be
    // using something like the standard deviation, but this works
    // for now.
    if (num_finished() != work.size()) return 0;

    return Now() - last_ping_time;
  }

  void assign_shard(TableMap& tables, int shard, bool should_service) {
    for (auto i : tables) {
      if (shard < i.second->num_shards()) {

        // Update local partition information, for performing put/fetches
        // on the master.
        PartitionInfo* p = i.second->shard_info(shard);
        p->set_owner(id);
        p->set_shard(shard);
        p->set_table(i.first);

        Taskid t(i.first, shard);
        if (should_service) {
          shards.insert(t);
        } else {
          shards.erase(shards.find(t));
        }
      }
    }
  }

  bool serves(Taskid id) const {
    return shards.find(id) != shards.end();
  }

  void assign_task(TaskState *s) {
    work[s->id] = s;
  }

  void remove_task(TaskState* s) {
    work.erase(work.find(s->id));
  }

  void clear_tasks() {
    work.clear();
  }

  void set_finished(const Taskid& id) {
    CHECK(work.find(id) != work.end());
    TaskState *t = work[id];
    CHECK(t->status == TaskState::ACTIVE);
    t->status = TaskState::FINISHED;
  }

  std::string str() {
    return StringPrintf("W: p: %d; a: %d a: %d f: %d", id, num_pending(),
        num_active(), num_assigned(), num_finished());
  }

#define COUNT_TASKS(name, type)\
  size_t num_ ## name() const {\
    int c = 0;\
    for (TaskMap::const_iterator i = work.begin(); i != work.end(); ++i)\
      if (i->second->status == TaskState::type) { ++c; }\
    return c;\
  }\
  int64_t name ## _size() const {\
      int64_t c = 0;\
      for (TaskMap::const_iterator i = work.begin(); i != work.end(); ++i)\
        if (i->second->status == TaskState::type) { c += i->second->size; }\
      return c;\
  }\
  vector<TaskState*> name() const {\
    vector<TaskState*> out;\
    for (TaskMap::const_iterator i = work.begin(); i != work.end(); ++i)\
      if (i->second->status == TaskState::type) { out.push_back(i->second); }\
    return out;\
  }

  COUNT_TASKS(pending, PENDING)
  COUNT_TASKS(active, ACTIVE)
  COUNT_TASKS(finished, FINISHED)
#undef COUNT_TASKS

  int num_assigned() const {
    return work.size();
  }
  int64_t total_size() const {
    int64_t out = 0;
    for (TaskMap::const_iterator i = work.begin(); i != work.end(); ++i) {
      out += 1 + i->second->size;
    }
    return out;
  }

  // Order pending tasks by our guess of how large they are
  bool get_next(const RunDescriptor& r, KernelRequest* msg) {
    vector<TaskState*> p = pending();

    if (p.empty()) {
      return false;
    }

    TaskState* best = *max_element(p.begin(), p.end(),
        &TaskState::WeightCompare);

    msg->set_kernel(r.kernel);
    msg->set_table(r.table->id());
    msg->set_shard(best->id.shard);
    for (auto i : r.args) {
      KV* kv = msg->add_args();
      kv->set_key(i.first);
      kv->set_value(i.second);
    }

    best->status = TaskState::ACTIVE;
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
    dead_workers.insert(strtod(bits[i].AsString().c_str(), NULL));
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
    if (workers_[i]->serves(Taskid(table, shard))) {
      return workers_[i];
    }
  }

  return NULL;
}

WorkerState* Master::assign_worker(int table, int shard) {
  WorkerState* ws = worker_for_shard(table, shard);
  int64_t work_size = 1;

  if (ws) {
    VLOG(1) << "Worker for shard: " << make_tuple(table, shard, ws->id);
    ws->assign_task(new TaskState(Taskid(table, shard), work_size));
    return ws;
  }

  WorkerState* best = NULL;
  for (size_t i = 0; i < workers_.size(); ++i) {
    WorkerState& w = *workers_[i];
    if (w.alive() && (best == NULL || w.shards.size() < best->shards.size())) {
      best = workers_[i];
    }
  }

  CHECK(best != NULL)
                         << "Ran out of workers!  Increase the number of partitions per worker!";

//  LOG(INFO) << "Assigned " << make_tuple(table, shard, best->id);
  CHECK(best->alive());

  VLOG(1) << "Assigning " << make_tuple(table, shard) << " to " << best->id;
  best->assign_shard(tables_, shard, true);
  return best;
}

void Master::send_table_assignments() {
  ShardAssignmentRequest req;

  for (size_t i = 0; i < workers_.size(); ++i) {
    WorkerState& w = *workers_[i];
    for (ShardSet::iterator j = w.shards.begin(); j != w.shards.end(); ++j) {
      ShardAssignment* s = req.add_assign();
      s->set_new_worker(i);
      s->set_table(j->table);
      s->set_shard(j->shard);
//      s->set_old_worker(-1);
    }
  }

  network_->SyncBroadcast(MessageTypes::SHARD_ASSIGNMENT, req);
}

bool Master::steal_work(const RunDescriptor& r, int idle_worker,
    double avg_completion_time) {
  if (!FLAGS_work_stealing) {
    return false;
  }

  WorkerState &dst = *workers_[idle_worker];

  if (!dst.alive()) {
    return false;
  }

  // Find the worker with the largest number of pending tasks.
  WorkerState& src = **max_element(workers_.begin(), workers_.end(),
      &WorkerState::PendingCompare);
  if (src.num_pending() == 0) {
    return false;
  }

  vector<TaskState*> pending = src.pending();

  TaskState *task = *max_element(pending.begin(), pending.end(),
      TaskState::WeightCompare);
  if (task->stolen) {
    return false;
  }

  double average_size = 0;

  for (int i = 0; i < r.table->num_shards(); ++i) {
    average_size += 1.0;
  }
  average_size /= r.table->num_shards();

  // Weight the cost of moving the table versus the time savings.
  double move_cost = std::max(1.0,
      2 * task->size * avg_completion_time / average_size);
  double eta = 0;
  for (size_t i = 0; i < pending.size(); ++i) {
    TaskState *p = pending[i];
    eta += std::max(1.0, p->size * avg_completion_time / average_size);
  }

//  LOG(INFO) << "ETA: " << eta << " move cost: " << move_cost;

  if (eta <= move_cost) {
    return false;
  }

  const Taskid& tid = task->id;
  task->stolen = true;

  LOG(INFO)<< "Worker " << idle_worker << " is stealing task "
  << make_tuple(tid.shard, task->size) << " from worker " << src.id;
  dst.assign_shard(tables_, tid.shard, true);
  src.assign_shard(tables_, tid.shard, false);

  src.remove_task(task);
  dst.assign_task(task);
  return true;
}

void Master::assign_shards(Table* t) {
  for (int j = 0; j < t->num_shards(); ++j) {
    assign_worker(t->id(), j);
  }

  send_table_assignments();
}

void Master::assign_tasks(const RunDescriptor& r, vector<int> shards) {
  for (size_t i = 0; i < workers_.size(); ++i) {
    WorkerState& w = *workers_[i];
    w.clear_tasks(); //XXX: did not delete task state, memory leak
  }

  for (size_t i = 0; i < shards.size(); ++i) {
    VLOG(1) << "Assigning worker for table " << r.table->id() << " for shard "
               << i << " of " << shards.size();
    assign_worker(r.table->id(), shards[i]);
  }
}

int Master::dispatch_work(const RunDescriptor& r) {
  int num_dispatched = 0;
  KernelRequest w_req;
  for (size_t i = 0; i < workers_.size(); ++i) {
    WorkerState& w = *workers_[i];
//    LOG(INFO)<< "Dispatching: " << w.str() << " : " << w_req;
    if (w.num_pending() > 0 && w.num_active() == 0) {
      w.get_next(r, &w_req);

      num_dispatched++;
      network_->Send(w.id + 1, MessageTypes::RUN_KERNEL, w_req);
    }
  }
  return num_dispatched;
}

void Master::dump_stats() {
  string status;
  for (int k = 0; k < num_workers(); ++k) {
    status += StringPrintf("%d/%d ", workers_[k]->num_finished(),
        workers_[k]->num_assigned());
  }
  LOG(INFO)<< StringPrintf("Running %s (%d); %s; assigned: %d done: %d",
      current_run_.kernel.c_str(), current_run_.shards.size(),
      status.c_str(), dispatched_, finished_);

}

int Master::reap_one_task() {
  MethodStats &mstats = method_stats_[ToString(current_run_.kernel)];
  KernelDone done_msg;
  int w_id = 0;

  if (network_->TryRead(rpc::ANY_SOURCE, MessageTypes::KERNEL_DONE, &done_msg,
      &w_id)) {
//    LOG(INFO) << "Done.";
    w_id -= 1;

    WorkerState& w = *workers_[w_id];

    Taskid task_id(done_msg.kernel().table(), done_msg.kernel().shard());
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

  dispatched_ = dispatch_work(current_run_);

  barrier();
}

void Master::barrier() {
  MethodStats &mstats = method_stats_[ToString(current_run_.kernel)];

  while (finished_ < current_run_.shards.size()) {
    PERIODIC(10, { DumpProfile(); dump_stats(); });

    dispatch_work(current_run_);

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
