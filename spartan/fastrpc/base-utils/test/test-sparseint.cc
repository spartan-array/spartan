#include <stdio.h>
#include <limits>

#include "base/all.h"

using namespace base;
using namespace std;

static void print_binary(const void* vbuf, int len) {
    char* buf = (char *) vbuf;
    for (int i = 0; i < len; i++) {
        for (int j = 0; j < 8; j++) {
            uint8_t u = buf[i];
            printf("%d", (u >> (7 - j)) & 0x1);
        }
        printf(" ");
    }
    printf("\n");
}

TEST(sparseint, dump_load_i32) {
    char buf[9];
    const i32 values[] = {0, 1, -1, -64, 63, 64, -65, -8192, 8191, -8193, 8192, -1048576, 1048575,
                    1048576, -1048577, -134217728, 134217727,
                    134217728, -134217729,
                    numeric_limits<i32>::max(), numeric_limits<i32>::min()};
    const int bs[] = {1, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3,
                    4, 4, 4, 4,
                    5, 5, 5, 5};
    for (size_t i = 0; i < arraysize(values); i++) {
        const i32 v = values[i];
        memset(buf, 0, 9);
        int bsize = SparseInt::dump(v, buf);
        print_binary(buf, bsize);
        print_binary(&v, 4);
        const i32 u = SparseInt::load_i32(buf);
        print_binary(&u, 4);
        Log::debug("%d -> bsize=%d -> %d", v, bsize, u);
        EXPECT_EQ(v, u);
        EXPECT_EQ(bsize, bs[i]);
    }
}

TEST(sparseint, dump_load_i64) {
    char buf[9];
    const i64 values[] = {0, 1, -1, -64, 63, 64, -65, -8192, 8191, -8193, 8192, -1048576, 1048575,
                    1048576, -1048577, -134217728, 134217727,
                    134217728, -134217729,
                    numeric_limits<i32>::max(), numeric_limits<i32>::min(),
                    -17179869184LL, 17179869183LL,
                    -17179869185LL, 17179869184LL,
                    -2199023255552LL, 2199023255551LL,
                    -2199023255553LL, 2199023255552LL,
                    -281474976710656LL, 281474976710655LL,
                    -281474976710657LL, 281474976710656LL,
                    -36028797018963968LL, 36028797018963967LL,
                    -36028797018963969LL, 36028797018963968LL,
                    numeric_limits<i64>::max(), numeric_limits<i64>::min()};
    const int bs[] = {1, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3,
                    4, 4, 4, 4,
                    5, 5, 5, 5,
                    5, 5, 6, 6,
                    6, 6, 7, 7,
                    7, 7, 8, 8,
                    8, 8, 9, 9,
                    9, 9};
    for (size_t i = 0; i < arraysize(values); i++) {
        const i64 v = values[i];
        memset(buf, 0, 9);
        int bsize = SparseInt::dump(v, buf);
        print_binary(buf, bsize);
        print_binary(&v, 8);
        const i64 u = SparseInt::load_i64(buf);
        print_binary(&u, 8);
        Log::debug("%lld -> bsize=%d -> %lld", v, bsize, u);
        EXPECT_EQ(v, u);
        EXPECT_EQ(bsize, bs[i]);
    }
}
