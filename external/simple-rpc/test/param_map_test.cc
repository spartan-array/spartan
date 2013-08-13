#include <string>

#include "param_map.h"
#include "test_unit.h"

namespace {

using std::string;
using base::ParamMap;

TEST(Basics, SimpleArgument) {
  const char* argv[] = { "SimpleArgument" , "--key=value" };
  ParamMap m;
  EXPECT_TRUE(m.parseArgv(2, argv));

  string value;
  EXPECT_TRUE(m.getParam("key", &value));
  EXPECT_EQ(value, "value");
}

TEST(Basics, ValueLessArgument) {
  const char* argv[] = { "ValueLessArgument" , "--valueless" };
  ParamMap m;
  EXPECT_TRUE(m.parseArgv(2, argv));

  string value("dummy value");
  EXPECT_TRUE(m.getParam("valueless", &value));
  EXPECT_EQ(value, "");
}

TEST(Basics, HelpArgument) {
  const char* argv[] = { "HelpArgument" , "--help" };
  ParamMap m;
  EXPECT_FALSE(m.parseArgv(2, argv));
}

TEST(Error, NameLessArgument) {
  const char* argv[] = { "NameLessArgument" , "--=value" };
  ParamMap m;
  EXPECT_FALSE(m.parseArgv(2, argv));
}

TEST(Error, InexistentArgument) {
  ParamMap m;
  string value;
  EXPECT_FALSE(m.getParam("inexistent", &value));
}

} // unnamed namespace

int main(int argc, char* argv[]) {
  return RUN_TESTS(argc, argv);
}
