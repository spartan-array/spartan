#include "test-common.h"
#include <vector>

using std::vector;
using std::pair;

class IterKernel: public Kernel {
public:
  void run() {
    auto t = get_typed<int, int>(this->table_id());
    for (int i = 0; i < 100; ++i) {
      t->update(i, i);
    }
  }
  DECLARE_REGISTRY_HELPER(Kernel, IterKernel);
};
DEFINE_REGISTRY_HELPER(Kernel, IterKernel);

int main(int argc, char** argv) {
  Master* m = start_cluster();

  auto t = m->create_table<int, int>(new Modulo<int>, NULL, new Replace<int>);
  m->map_shards(t, "IterKernel");

  // Merged iterator.
  auto i = t->get_iterator();
  vector < pair<int, int> > results;

  while (!i->done()) {
    results.push_back(make_pair(i->key(), i->value()));
    i->next();
  }

  sort(results.begin(), results.end());

  for (int i = 0; i < 100; ++i) {
    CHECK(results[i].first == i);
    CHECK(results[i].second == i);
  }
}
