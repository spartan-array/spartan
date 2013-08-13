#ifndef MASTER_H_
#define MASTER_H_

#include "sparrow/kernel.h"
#include "sparrow/table.h"
#include "sparrow/util/common.h"
#include "sparrow/util/timer.h"

#include "sparrow/sparrow_service.h"

#include <vector>
#include <map>

namespace sparrow {

class WorkerState;
class TaskState;

struct RunDescriptor {
  Table *table;

  string kernel;
  Kernel::ArgMap args;
  std::vector<int> shards;
};

typedef std::pair<int, int> ShardId;
class TaskState {
public:
  TaskState() :
      size(-1), stolen(false) {

  }
  TaskState(ShardId id, int64_t size) :
      id(id), size(size), stolen(false) {
  }

  ShardId id;
  int size;
  bool stolen;
};

typedef std::map<ShardId, TaskState> TaskMap;
typedef std::set<ShardId> ShardSet;

class WorkerState: private boost::noncopyable {
public:
  TaskMap pending;
  TaskMap active;
  TaskMap finished;

  // Table shards this worker is responsible for serving.
  ShardSet shards;

  int status;
  int id;

  double last_ping_time;
  double total_runtime;

  bool alive;

  WorkerProxy *proxy;
  HostPort addr;

  WorkerState(int w_id, HostPort addr) :
      id(w_id) {
    last_ping_time = Now();
    total_runtime = 0;
    alive = true;
    status = 0;
    this->addr = addr;
  }

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
  bool get_next(const RunDescriptor& r, RunKernelReq* msg) {
    if (pending.empty()) {
      return false;
    }

    TaskState state = pending.begin()->second;
    active[state.id] = state;
    pending.erase(pending.begin());

    msg->kernel = r.kernel;
    msg->table = r.table->id();
    msg->shard = state.id.second;
    msg->args = r.args;

    return true;
  }
};

Master* start_master(int port, int num_workers);

class Master: public TableContext, public MasterService {
public:
  Master(rpc::PollMgr* poller, int num_workers);
  ~Master();

  void wait_for_workers();

  // TableHelper implementation
  int id() const {
    return -1;
  }

  void flush();

  template<class K, class V>
  TableT<K, V>* create_table(SharderT<K>* sharder = new Modulo<K>(),
      AccumulatorT<V>* accum = new Replace<V>(), SelectorT<K, V>* selector =
          NULL, std::string sharder_opts = "", std::string accum_opts = "",
      std::string selector_opts = "") {

    TableT<K, V>* t = new TableT<K, V>();

    CreateTableReq req;
    int table_id = tables_.size();
    req.table_type = t->type_id();
    req.id = table_id;
    req.num_shards = workers_.size() * 2;

    req.accum.type_id = accum->type_id();
    req.accum.opts = accum_opts;

    req.sharder.type_id = sharder->type_id();
    req.sharder.opts = sharder_opts;

    if (selector != NULL) {
      req.selector.type_id = selector->type_id();
      req.selector.opts = selector_opts;
    } else {
      req.selector.type_id = -1;
    }

    sharder->init(sharder_opts);
    accum->init(accum_opts);

    t->init(table_id, req.num_shards);
    t->sharder = sharder;
    t->accum = accum;
    t->selector = selector;
    t->flush_frequency = 100;

    t->workers.resize(workers_.size());
    for (auto w : workers_) {
      t->workers[w->id] = w->proxy;
    }

    t->set_ctx(this);

    tables_[t->id()] = t;

    rpc::FutureGroup futures;
    for (auto w : workers_) {
      futures.add(w->proxy->async_create_table(req));
    }

    assign_shards(t);
    return t;
  }

  void map_shards(Table* t, const std::string& kernel) {
    RunDescriptor r;
    r.kernel = kernel;
    r.table = t;
    r.shards = range(0, t->num_shards());

    run(r);
  }

  void register_worker(const RegisterReq& req);
  void run(RunDescriptor r);
private:

// Find a worker to run a kernel on the given table and shard.  If a worker
// already serves the given shard, return it.  Otherwise, find an eligible
// worker and assign it to them.
  WorkerState* assign_shard(int table, int shard);

  void send_table_assignments();
  void assign_shards(Table *t);
  void assign_tasks(const RunDescriptor& r, std::vector<int> shards);
  int dispatch_work(const RunDescriptor& r);

  RunDescriptor current_run_;
  double current_run_start_;

  int num_workers_;
  std::vector<WorkerState*> workers_;
  std::set<rpc::Future*> running_kernels_;

  rpc::PollMgr *poller_;
  TableMap tables_;
  Timer runtime_;
};

}

#endif /* MASTER_H_ */
