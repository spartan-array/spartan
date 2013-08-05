#include "sparrow/master.h"
#include "sparrow/worker.h"
#include "sparrow/util/common.h"

using namespace sparrow;

class AccumKernel: public Kernel {
public:
  void run() {
    LOG(INFO) << "Table: " << table_id() << " : " << shard_id();
    auto t = get_typed<int, int>(this->table_id());
    for (int i = 0; i < 100; ++i) {
      t->update(0, i);
    }
  }
};
REGISTER_KERNEL(AccumKernel);

int main(int argc, char** argv) {
  sparrow::Init(argc, argv);

  if (!StartWorker()) {
    Master m;

    {
      auto t = m.create_table<int, int>(new Modulo<int>, new Min<int>);
      m.map_shards(t, "AccumKernel");
      LOG(INFO) << "Master fetch: " << t->get(0);
    }

    {
      auto t = m.create_table<int, int>(new Modulo<int>, new Max<int>);
      m.map_shards(t, "AccumKernel");
      LOG(INFO) << "Master fetch: " << t->get(0);
    }

    {
      auto t = m.create_table<int, int>(new Modulo<int>, new Sum<int>);
      m.map_shards(t, "AccumKernel");
      LOG(INFO) << "Master fetch: " << t->get(0);
    }

    {
      auto t = m.create_table<int, int>(new Modulo<int>, new Sum<int>);
      m.map_shards(t, "AccumKernel");
      LOG(INFO) << "Master fetch: " << t->get(0);
    }
  }
}
