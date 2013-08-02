#include "sparrow/master.h"
#include "sparrow/worker.h"
#include "sparrow/util/common.h"

using namespace sparrow;

class AccumKernel: public Kernel {
public:
  void run() {
    LOG(INFO) << "Table: " << table_id() << " : " << shard_id();
    Table* t = get_table(this->table_id());
    for (int i = 0; i < 100; ++i) {
      t->update(prim_to_string(0), prim_to_string(i));
    }
  }
};
REGISTER_KERNEL(AccumKernel);

int main(int argc, char** argv) {
  sparrow::Init(argc, argv);

  if (!StartWorker()) {
    Master m;

    {
      Table* t = m.create_table("Modulo", "intMin");
      m.map_shards(t, "AccumKernel");
      LOG(INFO)<< "Master fetch: " << string_to_prim<int>(t->get(prim_to_string(0)));
    }

    {
      Table* t = m.create_table("Modulo", "intMax");
      m.map_shards(t, "AccumKernel");
      LOG(INFO)<< "Master fetch: " << string_to_prim<int>(t->get(prim_to_string(0)));
    }

    {
      Table* t = m.create_table("Modulo", "intSum");
      m.map_shards(t, "AccumKernel");
      LOG(INFO)<< "Master fetch: " << string_to_prim<int>(t->get(prim_to_string(0)));
    }
  }
}
