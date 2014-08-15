#include <stdio.h>

#include "unittest.h"
#include "logging.h"
#include "strop.h"

namespace base {

void TestCase::fail() {
    failures_++;
}

TestMgr* TestMgr::instance_s = nullptr;

TestMgr* TestMgr::instance() {
    if (instance_s == nullptr) {
        instance_s = new TestMgr;
    }
    return instance_s;
}

TestCase* TestMgr::reg(TestCase* t) {
    tests_.push_back(t);
    return t;
}

void TestMgr::matched_tests(const char* match, std::vector<TestCase*>* matched) {
    std::vector<std::string>&& split = strsplit(match, ',');
    matched->clear();
    for (auto& t: tests_) {
        for (auto& s: split) {
            if (s.find('/') != std::string::npos) {
                if (t->group() + std::string("/") + t->name() == s) {
                    matched->push_back(t);
                }
            } else {
                if (t->group() == s) {
                    matched->push_back(t);
                }
            }
        }
    }
}

int TestMgr::parse_args(int argc, char* argv[], bool* show_help, bool* list_tests, std::vector<TestCase*>* selected) {
    *show_help = false;
    *list_tests = false;
    char* select = nullptr;
    char* skip = nullptr;
    std::vector<TestCase*> match;
    for (int i = 1; i < argc; i++) {
        if (streq(argv[i], "-h") || streq(argv[i], "--help")) {
            *show_help = true;
        } else if (streq(argv[i], "-l") || streq(argv[i], "--list")) {
            *list_tests = true;
        } else if (startswith(argv[i], "--select=")) {
            select = argv[i] + strlen("--select=");
            matched_tests(select, &match);
        } else if (startswith(argv[i], "--skip=")) {
            skip = argv[i] + strlen("--skip=");
            matched_tests(skip, &match);
        } else {
            return 1;
        }
    }
    if (select == nullptr && skip == nullptr) {
        *selected = tests_;
    } else if (select != nullptr && skip == nullptr) {
        *selected = match;
    } else if (select == nullptr && skip != nullptr) {
        selected->clear();
        for (auto& t: tests_) {
            bool select_me = true;
            for (auto& m: match) {
                if (t == m) {
                    select_me = false;
                }
            }
            if (select_me) {
                selected->push_back(t);
            }
        }
    } else { // select != nullptr && skip != nullptr
        printf("please provide either --select or --skip, not both\n");
        return 1;
    }
    return 0;
}

int TestMgr::run(int argc, char* argv[]) {
    bool show_help;
    bool list_tests;
    std::vector<TestCase*> selected;
    int r = parse_args(argc, argv, &show_help, &list_tests, &selected);
    if (r != 0 || show_help) {
        printf("usage: %s [-h|--help] [-l|--list] [--select,skip=group_x/test_y,group_z]\n", argv[0]);
        return r;
    }
    if (list_tests) {
        for (auto& t : selected) {
            printf("%s/%s\n", t->group(), t->name());
        }
        return r;
    }

    int failures = 0;
    int passed = 0;
    if (selected.size() > 0) {
        Log::info("The following %d test cases will be checked:", selected.size());
        for (auto& t : selected) {
            Log::info("    %s/%s", t->group(), t->name());
        }
    }
    for (auto& t : selected) {
        Log::info("--> starting test: %s/%s", t->group(), t->name());
        t->run();
        failures += t->failures();
        if (t->failures() == 0) {
            Log::info("<-- passed test: %s/%s", t->group(), t->name());
            passed++;
        } else {
            Log::error("X-- failed test: %s/%s", t->group(), t->name());
        }
    }
    Log::info("%d/%lu passed, %d failures\n", passed, selected.size(), failures);
    // cleanup testcases
    for (auto& t : tests_) {
        delete t;
    }
    delete this;
    return failures;
}

} // namespace base
