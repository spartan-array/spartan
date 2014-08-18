#include <map>

#include "base/all.h"
#include "rpc/utils.h"

using namespace base;
using namespace rpc;
using namespace std;

class EchoOnCopy {
    int val_;
public:
    EchoOnCopy(int v = 0): val_(v) {
        Log::info("    > default ctor");
    }
    EchoOnCopy(const EchoOnCopy& e): val_(e.val_) {
        Log::info("    > copy in const ctor");
    }
    const EchoOnCopy& operator =(const EchoOnCopy& e) {
        if (&e != this) {
            Log::info("    > copy in operator =");
            val_ = e.val_;
        } else {
            Log::info("    > self copy");
        }
        return *this;
    }
    bool operator <(const EchoOnCopy& e) const {
        return val_ < e.val_;
    }
};

TEST(utils, insert_into_map) {

    EchoOnCopy a(1987);
    EchoOnCopy b(1988);
    EchoOnCopy c(1989);

    const EchoOnCopy const_a(1987);
    const EchoOnCopy const_b(1988);
    const EchoOnCopy const_c(1989);

    map<EchoOnCopy, int> dict1;

    Log::info("dict1 <EchoOnCopy, int>");
    Log::info("  insert(make_pair)");
    dict1.insert(make_pair(a, 1987));
    Log::info("  insert_into_map()");
    insert_into_map(dict1, b, 1988);
    Log::info("  assign=");
    dict1[c] = 1989;

    map<EchoOnCopy, int> dict2;
    Log::info("dict2 <const EchoOnCopy, int>");
    Log::info("  insert(make_pair)");
    dict2.insert(make_pair(const_a, 1987));
    Log::info("  insert_into_map()");
    insert_into_map(dict2, const_b, 1988);
    Log::info("  assign=");
    dict2[const_c] = 1989;

    map<int, EchoOnCopy> dict3;
    Log::info("dict3 <int, EchoOnCopy>");
    Log::info("  insert(make_pair)");
    dict3.insert(make_pair(1987, a));
    Log::info("  insert_into_map()");
    insert_into_map(dict3, 1988, b);
    Log::info("  assign=");
    dict3[1989] = c;

    map<int, EchoOnCopy> dict4;
    Log::info("dict4 <int, EchoOnCopy>");
    Log::info("  insert(make_pair)");
    dict4.insert(make_pair(1987, const_a));
    Log::info("  insert_into_map()");
    insert_into_map(dict4, 1988, const_b);
    Log::info("  assign=");
    dict4[1989] = const_c;
}
