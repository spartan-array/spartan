#ifndef WORKER_H_
#define WORKER_H_

#include "sparrow/util/common.h"
#include "sparrow/util/rpc.h"
#include "sparrow/kernel.h"
#include "sparrow/table.h"

#include "sparrow/sparrow.pb.h"

#include <boost/thread.hpp>
#include <map>
#include <mpi.h>
#include <boost/unordered_map.hpp>
#include <boost/unordered_set.hpp>

using boost::shared_ptr;

namespace sparrow {

// If this node is the master, return false immediately.  Otherwise
// start a worker and exit when the computation is finished.
bool StartWorker();

class Worker: public TableHelper, private boost::noncopyable {
  struct Stub;
public:
  Worker(const ConfigData &c);
  virtual ~Worker();

  void Run();

  void KernelLoop();
  void TableLoop();
  Stats get_stats() {
    return stats_;
  }

  void CheckForMasterUpdates();
  void CheckNetwork();

  void create_table(const CreateTableRequest& req, EmptyMessage* resp, const rpc::RPCInfo& rpc);
  void get(const HashGet& get_req, TableData *get_resp,
                        const rpc::RPCInfo& rpc);
  void iterator_request(const IteratorRequest& iterator_req,
                             IteratorResponse *iterator_resp,
                             const rpc::RPCInfo& rpc);
  void assign_shards(const ShardAssignmentRequest& req,
                             EmptyMessage *resp, const rpc::RPCInfo& rpc);

  void handle_put_request();

  // Barrier: wait until all table data is transmitted.
  void flush(const EmptyMessage& req, FlushResponse *resp,
                   const rpc::RPCInfo& rpc);
  void apply(const EmptyMessage& req, EmptyMessage *resp,
                   const rpc::RPCInfo& rpc);
  void HandleStartCheckpointAsync(const CheckpointRequest& req,
                                  EmptyMessage* resp, const rpc::RPCInfo& rpc);
  void HandleFinishCheckpointAsync(const CheckpointFinishRequest& req,
                                   EmptyMessage *resp, const rpc::RPCInfo& rpc);
  void restore(const StartRestore& req, EmptyMessage *resp,
                          const rpc::RPCInfo& rpc);

  int peer_for_shard(int table_id, int shard) const;
  int id() const {
    return config_.worker_id();
  }
  ;
  int epoch() const {
    return epoch_;
  }

  int64_t pending_writes() const;
  bool network_idle() const;

  bool has_incoming_data() const;

  TableMap& tables() {
    return tables_;
  }

private:
  void StartCheckpoint(int epoch, CheckpointType type, bool deltaOnly);
  void FinishCheckpoint(bool deltaOnly);
  void UpdateEpoch(int peer, int peer_epoch);
  void UpdateEpochContinuous(int peer, int peer_epoch);

  mutable boost::recursive_mutex state_lock_;

  // The current epoch this worker is running within.
  int epoch_;

  int num_peers_;
  bool running_;
  bool kernel_active_;
  bool handling_putreqs_;
  CheckpointType active_checkpoint_;

  typedef boost::unordered_map<int, bool> CheckpointMap;
  CheckpointMap checkpoint_tables_;

  ConfigData config_;

  // The status of other workers.
  std::vector<Stub*> peers_;

  rpc::NetworkThread *network_;
  boost::unordered_set<Table*> dirty_tables_;

  uint32_t iterator_id_;
  boost::unordered_map<uint32_t, TableIterator*> iterators_;

  TableMap tables_;

  struct KernelId {
    string id_;
    int table_;
    int shard_;

    KernelId(string id, int table, int shard) :
        id_(id), table_(table), shard_(shard) {
    }

#define CMP_LESS(a, b, member)\
  if ((a).member < (b).member) { return true; }\
  if ((b).member < (a).member) { return false; }

    bool operator<(const KernelId& o) const {
      CMP_LESS(*this, o, id_);
      CMP_LESS(*this, o, table_);
      CMP_LESS(*this, o, shard_);
      return false;
    }
  };

  std::map<KernelId, Kernel*> kernels_;

  Stats stats_;
};

}

#endif /* WORKER_H_ */
