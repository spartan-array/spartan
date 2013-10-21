#include "spartan/worker.h"
#include "spartan/kernel.h"

namespace spartan {
Table* Kernel::get_table(int id) {
  return w_->tables()[id];
}

}
