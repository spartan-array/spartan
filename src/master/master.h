#ifndef MASTER_H_
#define MASTER_H_

#include "kernel/kernel.h"
#include "kernel/table.h"
#include "util/common.h"
#include "util/rpc.h"
#include "sparrow.pb.h"

#include <vector>
#include <map>

namespace sparrow {

class WorkerState;
class TaskState;

struct RunDescriptor {
  string kernel;
  string method;

  Table *table;
  std::vector<int> shards;
};

class Master: public TableHelper {
public:
  Master(const ConfigData &conf);
  ~Master();

  //TableHelper methods
  int id() const {
    return -1;
  }
  int epoch() const {
    return kernel_epoch_;
  }

  void run(RunDescriptor r);

  int peer_for_shard(int table, int shard) const {
    return tables_[table]->worker_for_shard(shard);
  }

  void handle_put_request() {
    LOG(FATAL)<< "Not implemented for master.";
  }

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
  WorkerState* assign_worker(int table, int shard);

  void send_table_assignments();
  bool steal_work(const RunDescriptor& r, int idle_worker, double avg_time);
  void assign_tables();
  void assign_tasks(const RunDescriptor& r, std::vector<int> shards);
  int dispatch_work(const RunDescriptor& r);

  void dump_stats();
  int reap_one_task();

  ConfigData config_;
  int checkpoint_epoch_;
  int kernel_epoch_;

  RunDescriptor current_run_;
  double current_run_start_;
  size_t dispatched_;//# of dispatched tasks
  size_t finished_;//# of finished tasks

  bool shards_assigned_;

  std::vector<WorkerState*> workers_;

  typedef std::map<string, MethodStats> MethodStatsMap;
  MethodStatsMap method_stats_;

  TableRegistry::Map& tables_;
  rpc::NetworkThread* network_;
  Timer runtime_;
};}

#endif /* MASTER_H_ */
