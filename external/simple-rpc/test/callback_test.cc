#include "rpc/callback.h"
#include "rpc/callback.h"
#include "test/test_unit.h"
#include "test/test_util.h"
#include "rpc/utils.h"

namespace {

using rpc::Callback;
using rpc::makeCallableOnce;
using rpc::makeCallable;
using test::Counter;

TEST(Once, Simple) {
  Counter c;
  Callback<void>* cb = makeCallableOnce(&Counter::inc, &c);
  EXPECT_TRUE(cb->once());
  (*cb)();
  EXPECT_EQ(c.count(), 1);
}

TEST(Once, Binding) {
  // early
  Counter c;
  Callback<void>* cb1 = makeCallableOnce(&Counter::incBy, &c, 2);
  EXPECT_TRUE(cb1->once());
  (*cb1)();
  EXPECT_EQ(c.count(), 2);

  // late
  c.reset();
  Callback<void, int>* cb2 = makeCallableOnce(&Counter::incBy, &c);
  EXPECT_TRUE(cb2->once());
  (*cb2)(3);
  EXPECT_EQ(c.count(), 3);
}

TEST(Once, Currying) {
  Counter c;
  Callback<void, int>* cb1 = makeCallableOnce(&Counter::incBy, &c);
  Callback<void>* cb2 =
    makeCallableOnce(&Callback<void,int>::operator(), cb1, 4);
  (*cb2)();
  EXPECT_EQ(c.count(), 4);
}

TEST(Once, ReturnType) {
  Counter c;
  c.set(7);
  Callback<bool, int, int>* cb1 = makeCallableOnce(&Counter::between, &c);
  EXPECT_TRUE((*cb1)(5, 10));

  Callback<bool, int>* cb2 = makeCallableOnce(&Counter::between, &c, 5);
  EXPECT_TRUE(cb2->once());
  EXPECT_TRUE((*cb2)(10));

  Callback<bool>* cb3 = makeCallableOnce(&Counter::between, &c, 5, 10);
  EXPECT_TRUE((*cb3)());
}

TEST(Many, Simple) {
  Counter c;
  Callback<void>* cb = makeCallable(&Counter::inc, &c);
  EXPECT_FALSE(cb->once());
  (*cb)();
  (*cb)();
  EXPECT_EQ(c.count(), 2);
  delete cb;
}

// For threadpool interface run()
TEST(Run, Simple) {
  Counter c;
  Callback<void>* cb = makeCallable(&Counter::inc, &c);
  EXPECT_FALSE(cb->once());
  cb->run();
  cb->run();
  EXPECT_EQ(c.count(), 2);
  delete cb;
}

TEST(TheadPool, Simple) {
  rpc::ThreadPool* pool = new rpc::ThreadPool(1);
  Counter c;
  pool->run_async([&] {c.incBy(2);});
  pool->release();
  EXPECT_EQ(c.count(), 2)
}

} // unnamed namespace

int main(int argc, char *argv[]) {
  return RUN_TESTS(argc, argv);
}
