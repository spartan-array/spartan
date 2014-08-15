#include "base/all.h"

using namespace base;

TEST(demo, expect_eq) {
    EXPECT_EQ(2, 2 + 0);
}

TEST(demo, expect_neq) {
    EXPECT_NEQ(1 + 2, 2);
}

/*
TEST(demo, will_fail) {
    EXPECT_TRUE(false);
    EXPECT_FALSE(true);
    EXPECT_EQ(3, 1 + 4);
    EXPECT_NEQ(3, 1 + 2);
}
*/

int main(int argc, char* argv[]) {
    return RUN_TESTS(argc, argv);
}
