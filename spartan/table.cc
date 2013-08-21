#include "spartan/table.h"
#include "spartan/util/registry.h"
#include "spartan/util/timer.h"

using namespace spartan;

namespace spartan {

static pthread_key_t ctx_key_ = 0;
static rpc::Mutex ctx_lock_;

TableContext* TableContext::get_context() {
  rpc::ScopedLock l(&ctx_lock_);
  CHECK(ctx_key_ != 0);
  auto result = (TableContext*)pthread_getspecific(ctx_key_);
  CHECK(result != NULL);
  return result;
}

void TableContext::set_context(TableContext* ctx) {
  rpc::ScopedLock l(&ctx_lock_);
  if (ctx_key_ == 0) {
    pthread_key_create(&ctx_key_, NULL);
  }
  pthread_setspecific(ctx_key_, ctx);
}

}
