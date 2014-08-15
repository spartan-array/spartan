#include "base/all.h"

using namespace base;

class IntRange: public Enumerator<int> {
public:
    IntRange(int last): cur_(0), last_(last) { }
    bool has_next() {
        return cur_ < last_;
    }
    int next() {
        int v = cur_;
        cur_++;
        return v;
    }
private:
    int cur_, last_;
};

TEST(enumer, int_range_enum) {
    IntRange rng(5);
    while (rng) {
        Log::debug("range enumerator: %d", rng());
    }
}

TEST(enumer, merged_enum) {
    IntRange src1(5);
    IntRange src2(15);
    MergedEnumerator<int> me;
    me.add_source(&src1);
    me.add_source(&src2);
    while (me) {
        Log::debug("merged enumerator: %d", me());
    }
}
