#include "spartan/master.h"
#include "spartan/worker.h"
#include "spartan/util/common.h"

using namespace spartan;
using namespace rpc;

Master* start_cluster() {
  auto master = start_master(9999, 4);
  sleep(1);

  for (int i = 0; i < 4; ++i) {
    start_worker(
        StringPrintf("%s:9999", get_host_name().c_str()).c_str(),
        10000 + i);
  }

  master->wait_for_workers();

  rpc::Log::set_level(rpc::Log::INFO);
  return master;
}
