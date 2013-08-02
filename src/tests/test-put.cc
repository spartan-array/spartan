#include "sparrow/master.h"
#include "sparrow/worker.h"
#include "sparrow/util/common.h"

using namespace sparrow;

class PutKernel: public Kernel {
public:
  void run() {
    auto t = get_table(0)->cast<string, string>();
    if (shard_id() == 0) {
      std::string binary_val("WorldBinary");
      for (size_t i = 0; i < 100; ++i) {
        binary_val.push_back(0);
      }

      t->put("Hello", "World.");
      t->put("HelloBinary", binary_val);

      for (size_t i = 0; i < 100; ++i) {
        t->put(StringPrintf("k_%d", i), "WTF");
      }
    }
  }
};

class GetKernel: public Kernel {
public:
  void run() {
    TableT<string, string>* t = get_table(0)->cast<string, string>();
    LOG(INFO)<< shard_id() << " : " << t->get("Hello");
  }
};

REGISTER_KERNEL(PutKernel);
REGISTER_KERNEL(GetKernel);

int main(int argc, char** argv) {
  sparrow::Init(argc, argv);

  if (!StartWorker()) {
    Master m;
    LOG(INFO)<< "here.";
    TableT<string, string>* t = m.create_table<string, string>();
    m.map_shards(t, "PutKernel");
    m.map_shards(t, "GetKernel");

    LOG(INFO)<< "Master: " << t->get("Hello");
    std::string binary_result = t->get("HelloBinary");
    LOG(INFO)<< binary_result.size();
  }
}
