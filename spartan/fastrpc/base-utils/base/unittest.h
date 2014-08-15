#pragma once

#include <vector>
#include <iostream>

namespace base {

class TestCase {
    const char* group_;
    const char* name_;
    int failures_;
public:
    TestCase(const char* _group, const char* _name): group_(_group), name_(_name), failures_(0) { }
    virtual ~TestCase() {}
    virtual void run() = 0;
    const char* group() { return group_; }
    const char* name() { return name_; }
    void reset() { failures_ = 0; }
    void fail();
    int failures() { return failures_; }
};

// singleton
class TestMgr {
    TestMgr() { }
    static TestMgr* instance_s;
    std::vector<TestCase*> tests_;
public:
    static TestMgr* instance();
    TestCase* reg(TestCase*);
    int parse_args(int argc, char* argv[], bool* show_help, bool* list_tests, std::vector<TestCase*>* selected);
    void matched_tests(const char* match, std::vector<TestCase*>* matched);
    int run(int argc, char* argv[]);
};

} // namespace base

#define TEST_CLASS_NAME(group, name) \
    TestCase_ ## group ## _ ## name

#define TEST(group, name) \
    class TEST_CLASS_NAME(group, name) : public ::base::TestCase { \
        static TestCase* me_s; \
    public: \
        TEST_CLASS_NAME(group, name)(); \
        void run(); \
    }; \
    TEST_CLASS_NAME(group, name)::TEST_CLASS_NAME(group, name)(): \
        TestCase(#group, #name) { } \
    ::base::TestCase* TEST_CLASS_NAME(group, name)::me_s = \
        ::base::TestMgr::instance()->reg(new TEST_CLASS_NAME(group, name)); \
    void TEST_CLASS_NAME(group, name)::run()

#define RUN_TESTS(argc, argv) ::base::TestMgr::instance()->run((argc), (argv));

#define EXPECT_TRUE(a) \
    { \
        auto va = (a); \
        if (!va) { \
            fail(); \
            std::cout << "    *** expected true: '" << #a << "', got false (" \
                << __FILE__ << ':' << __LINE__  << ')' << std::endl; \
        } \
    }

#define EXPECT_FALSE(a) \
    { \
        auto va = (a); \
        if (va) { \
            fail(); \
            std::cout << "    *** expected false: '" << #a << "', got true (" \
                << __FILE__ << ':' << __LINE__  << ')' << std::endl; \
        } \
    }

#define EXPECT_BINARY_OP_GENERATOR(op, a, b) \
    { \
        auto va = (a); \
        auto vb = (b); \
        if (!(va op vb)) { \
            fail(); \
            std::cout << "    *** expected: '" \
                << #a << ' ' << #op << ' ' << #b \
                << "', got " << va << " and " << vb \
                << " (" << __FILE__ << ':' << __LINE__ << ')' << std::endl; \
        } \
    }

#define EXPECT_LT(a, b) EXPECT_BINARY_OP_GENERATOR(<, (a), (b))
#define EXPECT_LE(a, b) EXPECT_BINARY_OP_GENERATOR(<=, (a), (b))
#define EXPECT_GT(a, b) EXPECT_BINARY_OP_GENERATOR(>, (a), (b))
#define EXPECT_GE(a, b) EXPECT_BINARY_OP_GENERATOR(>=, (a), (b))
#define EXPECT_EQ(a, b) EXPECT_BINARY_OP_GENERATOR(==, (a), (b))
#define EXPECT_NEQ(a, b) EXPECT_BINARY_OP_GENERATOR(!=, (a), (b))
