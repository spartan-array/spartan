#include "spartan/master.h"
#include "spartan/worker.h"
#include "spartan/util/common.h"

using namespace spartan;
using namespace rpc;

Master* start_cluster() {
  auto master = start_master(9999, 4);

  for (int i = 0; i < 4; ++i) {
    start_worker("localhost:9999", 10000 + i);
  }

  master->wait_for_workers();
  return master;
}
