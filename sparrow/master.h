#ifndef MASTER_H_
#define MASTER_H_

#include "sparrow/kernel.h"
#include "sparrow/table.h"
#include "sparrow/util/common.h"
#include "sparrow/util/rpc.h"
#include "sparrow/util/timer.h"
#include "sparrow/sparrow.pb.h"

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

class Master: public TableContext {
public:
  Master();
  ~Master();

  // TableHelper implementation
  int id() const {
    return -1;
  }

  int epoch() const {
    return kernel_epoch_;
  }

  int peer_for_shard(int table, int shard) const {
    return tables_.find(table)->second->worker_for_shard(shard);
  }

  void flush_network();

  int num_workers() const {
    return network_->size() - 1;
  }

  template<class K, class V>
  TableT<K, V>* create_table(
      SharderT<K>* sharder = new Modulo<K>(),
      AccumulatorT<V>* accum = new Replace<V>(),
      SelectorT<K, V>* selector = NULL,
      std::string sharder_opts = "",
      std::string accum_opts = "",
      std::string selector_opts = "") {

    TableT<K, V>* t = new TableT<K, V>();

    CreateTableRequest req;
    int table_id = tables_.size();
    req.set_table_type(t->type_id());
    req.set_id(table_id);
    req.set_num_shards(10);
    req.set_accum_type(accum->type_id());
    req.set_sharder_type(sharder->type_id());
    req.set_sharder_opts(sharder_opts);
    req.set_accum_opts(accum_opts);
    req.set_selector_opts(selector_opts);

    if (selector != NULL) {
      req.set_selector_type(selector->type_id());
    }

    sharder->init(sharder_opts);
    accum->init(accum_opts);

    t->init(table_id, req.num_shards());
    t->sharder = sharder;
    t->accum = accum;
    t->set_ctx(this);

    tables_[t->id()] = t;

    network_->SyncBroadcast(MessageTypes::CREATE_TABLE, req);

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

  void run(RunDescriptor r);

  void barrier();

// Blocking.  Instruct workers to save table and kernel state.
// When this call returns, all requested tables in the system will have been
// committed to disk.
  void checkpoint();

// Attempt restore from a previous checkpoint for this job.  If none exists,
// the process is left in the original state, and this function returns false.
  bool restore();

private:
  void start_checkpoint();
  void finish_checkpoint();

  WorkerState* worker_for_shard(int table, int shard);

// Find a worker to run a kernel on the given table and shard.  If a worker
// already serves the given shard, return it.  Otherwise, find an eligible
// worker and assign it to them.
  WorkerState* assign_shard(int table, int shard);

  void send_table_assignments();
  void assign_shards(Table *t);
  void assign_tasks(const RunDescriptor& r, std::vector<int> shards);
  int dispatch_work(const RunDescriptor& r);

  void dump_stats();
  int reap_one_task();

  ConfigData config_;
  int checkpoint_epoch_;
  int kernel_epoch_;

  RunDescriptor current_run_;
  double current_run_start_;
  size_t dispatched_; //# of dispatched tasks
  size_t finished_; //# of finished tasks

  bool shards_assigned_;

  std::vector<WorkerState*> workers_;

  typedef std::map<string, MethodStats> MethodStatsMap;
  MethodStatsMap method_stats_;

  TableMap tables_;
  rpc::NetworkThread* network_;
  Timer runtime_;
};

}

#endif /* MASTER_H_ */
