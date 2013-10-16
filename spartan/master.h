#ifndef MASTER_H_
#define MASTER_H_

#include "spartan/kernel.h"
#include "spartan/table.h"
#include "spartan/util/common.h"
#include "spartan/util/stringpiece.h"
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

  ShardId() :
      table(-1), shard(-1) {

  }
  ShardId(int t, int s) :
      table(t), shard(s) {
  }

  bool operator<(const ShardId& r) const {
    if (table < r.table) {
      return true;
    }
    if (table > r.table) {
      return false;
    }
    if (shard < r.shard) {
      return true;
    }
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

  int num_workers() {
    return num_workers_;
  }

  Table* create_table(Sharder* sharder = NULL, Accumulator* combiner = NULL,
      Accumulator* reducer = NULL, Selector* selector = NULL);

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
