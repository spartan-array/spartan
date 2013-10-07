#ifndef WORKER_H_
#define WORKER_H_

#include <boost/unordered_map.hpp>
#include <boost/unordered_set.hpp>
#include <boost/noncopyable.hpp>
#include <map>

#include "spartan/util/common.h"
#include "spartan/kernel.h"
#include "spartan/table.h"
#include "spartan/spartan_service.h"

namespace spartan {

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
  void destroy_table(const rpc::i32& id);
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

  Table* get_table(int id) const {
    return tables_.find(id)->second;
  }

  void wait_for_shutdown();

  void set_server(rpc::Server* s) {
    server_ = s;
  }

private:
  rpc::CondVar running_cv_;
  bool running_;
  rpc::Mutex lock_;

  int id_;

  ConfigData config_;

  // The status of other workers.
  std::vector<WorkerProxy*> peers_;
  rpc::PollMgr* poller_;

  uint32_t current_iterator_id_;
  boost::unordered_map<uint32_t, TableIterator*> iterators_;

  TableMap tables_;
  rpc::Server* server_;
};

}

#endif /* WORKER_H_ */
