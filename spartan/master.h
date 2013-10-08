#ifndef MASTER_H_
#define MASTER_H_

#include "spartan/kernel.h"
#include "spartan/table.h"
#include "spartan/util/common.h"
#include "spartan/util/timer.h"
#include "spartan/spartan_service.h"

#include "rpc/utils.h"

#include <vector>
#include <map>
#include <boost/noncopyable.hpp>

namespace spartan {

class WorkerState;
class TaskState;

struct ShardId {
  int table;
  int shard;

  ShardId() : table(-1), shard(-1) {

  }
  ShardId(int t, int s) :
      table(t), shard(s) {
  }

  bool operator<(const ShardId& r) const {
    if (table < r.table) { return true; }
    if (table > r.table) { return false; }
    if (shard < r.shard) { return true; }
    return false;
  }
};

class TaskState {
public:
  TaskState() :
      id(-1, -1), size(-1) {
  }

  TaskState(ShardId id, int64_t size, ArgMap args) :
      id(id), size(size), args(args) {
  }

  ShardId id;
  int size;
  ArgMap args;
};

struct WorkItem {
  ArgMap args;
  ShardId locality;
};

typedef std::vector<WorkItem> WorkList;
typedef std::multimap<ShardId, TaskState> TaskMap;
typedef std::set<ShardId> ShardSet;
Master* start_master(int port, int num_workers);

int worker_id(WorkerState*);
WorkerProxy* worker_proxy(WorkerState*);

class Master: public TableContext, public MasterService {
public:
  Master(int num_workers);
  ~Master();

  void wait_for_workers();

  // TableHelper implementation
  int id() const {
    return -1;
  }

  void shutdown();

  void flush();

  void destroy_table(int table_id);

  void destroy_table(Table* t) {
    destroy_table(t->id());
  }

  int num_workers() {
    return num_workers_;
  }

  template<class K, class V>
  TableT<K, V>* create_table(SharderT<K>* sharder = new Modulo<K>(),
      AccumulatorT<K, V>* combiner = NULL,
      AccumulatorT<K, V>* reducer = NULL,
      SelectorT<K, V>* selector = NULL) {
    Timer timer;
    wait_for_workers();

    TableT<K, V>* t = new TableT<K, V>();

    // Crash here if we can't find the sharder/accumulators.
    delete TypeRegistry<Sharder>::get_by_id(sharder->type_id());

    CreateTableReq req;
    int table_id = table_id_counter_++;

    Log_debug("Creating table %d", table_id);
    req.table_type = t->type_id();
    req.id = table_id;
    req.num_shards = workers_.size() * 2 + 1;

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

    req.sharder.type_id = sharder->type_id();
    req.sharder.opts = sharder->opts();

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

    // Log_info("Table created in %f seconds", timer.elapsed());
    return t;
  }

  void map_shards(Table* t, const std::string& kernel) {
    map_shards(t, TypeRegistry<Kernel>::get_by_name(kernel));
  }

  void map_shards(Table* t, Kernel* k);

  void map_worklist(WorkList worklist, Kernel* k);

  Table* get_table(int id) const {
    return tables_.find(id)->second;
  }

  void set_server(rpc::Server* s) {
    server_ = s;
  }

private:
  void wait_for_completion(Kernel* k);
  void register_worker(const RegisterReq& req);

  // Find a worker to run a kernel on the given table and shard.  If a worker
  // already serves the given shard, return it.  Otherwise, find an eligible
  // worker and assign it to them.
  WorkerState* assign_shard(int table, int shard);

  void send_table_assignments();
  void assign_shards(Table *t);
  void assign_tasks(Table* t, std::vector<int> shards);
  int dispatch_work(Kernel* k);
  int num_pending();

  int num_workers_;
  std::vector<WorkerState*> workers_;

  rpc::Mutex lock_;
  std::map<int, rpc::Future*> running_kernels_;

  rpc::PollMgr *client_poller_;
  TableMap tables_;
  Timer runtime_;

  bool initialized_;
  int table_id_counter_;

  rpc::Server* server_;
};

}

#endif /* MASTER_H_ */
