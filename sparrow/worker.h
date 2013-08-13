#ifndef WORKER_H_
#define WORKER_H_

#include <boost/unordered_map.hpp>
#include <boost/unordered_set.hpp>
#include <map>

#include "sparrow/util/common.h"
#include "sparrow/kernel.h"
#include "sparrow/table.h"
#include "sparrow/sparrow_service.h"

namespace sparrow {

Worker* start_worker(const std::string& master, int port = -1);

class Worker: public TableContext,
    public WorkerService,
    private boost::noncopyable {
  struct Stub;
public:
  Worker(rpc::PollMgr* poller);
  ~Worker();

  void initialize(const WorkerInitReq& req);
  void create_table(const CreateTableReq& req);
  void delete_table(const DeleteTableReq& req);
  void get(const GetRequest& req, TableData* resp);
  void assign_shards(const ShardAssignmentReq& req);
  void run_kernel(const RunKernelReq& req);
  void get_iterator(const IteratorReq& req, IteratorResp* resp);
  void put(const TableData& req);
  void flush();
  void shutdown();

  int id() const {
    return id_;
  }

  TableMap& tables() {
    return tables_;
  }

private:
  mutable boost::recursive_mutex state_lock_;

  int id_;
  bool running_;
  bool kernel_active_;
  bool handling_putreqs_;

  ConfigData config_;

  // The status of other workers.
  std::vector<WorkerProxy*> peers_;
  rpc::PollMgr* poller_;

  uint32_t current_iterator_id_;
  boost::unordered_map<uint32_t, TableIterator*> iterators_;

  TableMap tables_;
};

}

#endif /* WORKER_H_ */
