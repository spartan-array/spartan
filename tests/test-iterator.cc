#include "sparrow/master.h"
#include "sparrow/worker.h"
#include "sparrow/util/common.h"

#include <vector>

using namespace sparrow;
using namespace std;

class IterKernel: public Kernel {
public:
  void run() {
    auto t = get_typed<int, int>(this->table_id());
    for (int i = 0; i < 100; ++i) {
      t->update(i, i);
    }
  }
};
REGISTER_KERNEL(IterKernel);

int main(int argc, char** argv) {
  sparrow::Init(argc, argv);

  if (!StartWorker()) {
    Master m;

    auto t = m.create_table<int, int>(new Modulo<int>, new Replace<int>);
    m.map_shards(t, "IterKernel");

    // Merged iterator.
    auto i = t->get_iterator();
    vector<pair<int, int> > results;

    while (!i->done()) {
      results.push_back(make_pair(i->key(), i->value()));
      i->next();
    }

    sort(results.begin(), results.end());

    for (int i = 0; i < 100; ++i) {
      CHECK_EQ(results[i].first, i);
      CHECK_EQ(results[i].second, i);
    }
  }
}
