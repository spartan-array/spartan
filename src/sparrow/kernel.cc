#include "sparrow/worker.h"
#include "sparrow/kernel.h"

namespace sparrow {
Table* Kernel::get_table(int id) {
  return w_->tables()[id];
}

}
