#include "test-common.h"

class AccumKernel: public Kernel {
public:
  void run() {
    auto t = get_typed<int, int>(this->table_id());
    for (int i = 0; i < 100; ++i) {
      t->update(0, i);
    }
  }
};
REGISTER_KERNEL(AccumKernel);

int main(int argc, char** argv) {
  Master *m = start_cluster();

  {
    auto t = m->create_table<int, int>(new Modulo<int>, new Min<int>);
    m->map_shards(t, "AccumKernel");
    Log_info("Master fetch: %d", t->get(0));
  }

  {
    auto t = m->create_table<int, int>(new Modulo<int>, new Max<int>);
    m->map_shards(t, "AccumKernel");
    Log_info("Master fetch: %d", t->get(0));
  }

  {
    auto t = m->create_table<int, int>(new Modulo<int>, new Sum<int>);
    m->map_shards(t, "AccumKernel");
    Log_info("Master fetch: %d", t->get(0));
  }

  {
    auto t = m->create_table<int, int>(new Modulo<int>, new Sum<int>);
    m->map_shards(t, "AccumKernel");
    Log_info("Master fetch: %d", t->get(0));
  }
}
