#include "test-common.h"

class AccumKernel: public Kernel {
public:
  void run() {
    auto t = get_typed<int, int>(this->table_id());
    auto s = this->shard_id();
    for (int i = 0; i < s; ++i) {
      t->update(0, i);
    }
    for (int i = 0; i < 1000; ++i) {
      t->update(100 + i, i);
    }
  }

  DECLARE_REGISTRY_HELPER(Kernel, AccumKernel);
};
DEFINE_REGISTRY_HELPER(Kernel, AccumKernel);

int main(int argc, char** argv) {
  Master *m = start_cluster();

  {
    auto t = m->create_table<int, int>(new Modulo<int>, NULL, new Min<int>);
    m->map_shards(t, "AccumKernel");
    Log_info("Master fetch: %d", t->get(0));
  }

  {
    auto t = m->create_table<int, int>(new Modulo<int>, new Max<int>, new Replace<int>);
    m->map_shards(t, "AccumKernel");
    Log_info("Master fetch: %d", t->get(0));
  }

  {
    auto t = m->create_table<int, int>(new Modulo<int>, NULL, new Max<int>);
    m->map_shards(t, "AccumKernel");
    Log_info("Master fetch: %d", t->get(0));
  }

  {
    auto t = m->create_table<int, int>(new Modulo<int>, NULL, new Replace<int>);
    m->map_shards(t, "AccumKernel");
    Log_info("Master fetch: %d", t->get(0));
  }

  {
    auto t = m->create_table<int, int>(new Modulo<int>, NULL, new Sum<int>);
    m->map_shards(t, "AccumKernel");
    Log_info("Master fetch: %d", t->get(0));
  }

  {
    auto t = m->create_table<int, int>(new Modulo<int>, new Sum<int>, new Sum<int>);
    m->map_shards(t, "AccumKernel");
    Log_info("Master fetch: %d", t->get(0));
  }
}
