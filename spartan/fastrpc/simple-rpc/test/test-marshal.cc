#include <unistd.h>
#include <fcntl.h>
#include <cstdlib>
#include <limits>

#include "base/all.h"
#include "rpc/marshal.h"

using namespace rpc;
using namespace std;

TEST(marshal, uint_types) {
    Marshal m;
    {
        uint8_t u = numeric_limits<uint8_t>::max();
        uint8_t v = 0;
        m << u;
        EXPECT_EQ(m.content_size(), sizeof(uint8_t));
        m >> v;
        EXPECT_EQ(u, v);
    }
    {
        uint16_t u = numeric_limits<uint16_t>::max();
        uint16_t v = 0;
        m << u;
        EXPECT_EQ(m.content_size(), sizeof(uint16_t));
        m >> v;
        EXPECT_EQ(u, v);
    }
    {
        uint32_t u = numeric_limits<uint32_t>::max();
        uint32_t v = 0;
        m << u;
        EXPECT_EQ(m.content_size(), sizeof(uint32_t));
        m >> v;
        EXPECT_EQ(u, v);
    }
    {
        uint64_t u = numeric_limits<uint64_t>::max();
        uint64_t v = 0;
        m << u;
        EXPECT_EQ(m.content_size(), sizeof(uint64_t));
        m >> v;
        EXPECT_EQ(u, v);
    }
}

TEST(marshal, content_size) {
    Marshal m;
    rpc::i32 a = 4;
    EXPECT_EQ(m.content_size(), 0u);
    m << a;
    EXPECT_EQ(m.content_size(), 4u);
    rpc::i32 b = 9;
    m >> b;
    EXPECT_EQ(m.content_size(), 0u);
    EXPECT_EQ(a, b);
    i8 c = -3;
    m << c;
    EXPECT_EQ(m.content_size(), 1u);
    i8 d;
    m >> d;
    EXPECT_EQ(m.content_size(), 0u);
    EXPECT_EQ(c, d);
    i16 e = -3;
    m << e;
    EXPECT_EQ(m.content_size(), 2u);
    i16 f;
    m >> f;
    EXPECT_EQ(m.content_size(), 0u);
    EXPECT_EQ(c, d);

    v32 a1 = -3;
    m << a1;
    EXPECT_EQ(m.content_size(), 1u);
    v32 b1;
    m >> b1;
    EXPECT_EQ(m.content_size(), 0u);
    EXPECT_EQ(a1.get(), b1.get());

    a1 = numeric_limits<i32>::max();
    m << a1;
    EXPECT_EQ(m.content_size(), 5u);
    m >> b1;
    EXPECT_EQ(m.content_size(), 0u);
    EXPECT_EQ(a1.get(), b1.get());

    a1 = numeric_limits<i32>::min();
    m << a1;
    EXPECT_EQ(m.content_size(), 5u);
    m >> b1;
    EXPECT_EQ(m.content_size(), 0u);
    EXPECT_EQ(a1.get(), b1.get());

    v64 a2 = -1987;
    m << a2;
    EXPECT_EQ(m.content_size(), 2u);
    v64 b2;
    m >> b2;
    EXPECT_EQ(m.content_size(), 0u);
    EXPECT_EQ(a2.get(), b2.get());

    a2 = numeric_limits<i64>::max();
    m << a2;
    EXPECT_EQ(m.content_size(), 9u);
    m >> b2;
    EXPECT_EQ(m.content_size(), 0u);
    EXPECT_EQ(a2.get(), b2.get());

    a2 = numeric_limits<i64>::min();
    m << a2;
    EXPECT_EQ(m.content_size(), 9u);
    m >> b2;
    EXPECT_EQ(m.content_size(), 0u);
    EXPECT_EQ(a2.get(), b2.get());
}


const i64 g_chunk_size = 1000;
const i64 g_bytes_per_writer = 1000 * 1000 * 1000;
Marshal g_mt_benchmark_marshal;
pthread_mutex_t g_mt_benchmark_mutex = PTHREAD_MUTEX_INITIALIZER;

static void* start_mt_benchmark_writers(void* args) {
    char* dummy_data = new char[g_chunk_size];
    memset(dummy_data, 0, g_chunk_size);
    int n_bytes_written = 0;
    while (n_bytes_written < g_bytes_per_writer) {
        int n_write = std::min(g_chunk_size, g_bytes_per_writer - n_bytes_written);
        Pthread_mutex_lock(&g_mt_benchmark_mutex);
        int ret = g_mt_benchmark_marshal.write(dummy_data, n_write);
        Pthread_mutex_unlock(&g_mt_benchmark_mutex);
        verify(ret > 0);
        n_bytes_written += ret;
    }
    delete[] dummy_data;
    pthread_exit(nullptr);
    return nullptr;
}

// multi-thread benchmark
TEST(marshal, mt_benchmark) {
    const int n_writers = 10;
    pthread_t th_writers[n_writers];

    for (int i = 0; i < n_writers; i++) {
        Pthread_create(&th_writers[i], nullptr, start_mt_benchmark_writers, nullptr);
    }

    int null_fd = open("/dev/null", O_WRONLY);
    i64 n_bytes_read = 0;
    double report_time = -1.0;
    Timer xfer_timer;
    xfer_timer.start();
    while (n_bytes_read < n_writers * g_bytes_per_writer) {
        Pthread_mutex_lock(&g_mt_benchmark_mutex);
        int ret = g_mt_benchmark_marshal.write_to_fd(null_fd);
        Pthread_mutex_unlock(&g_mt_benchmark_mutex);
        verify(ret >= 0);
        n_bytes_read += ret;

        struct timeval tm;
        gettimeofday(&tm, nullptr);
        double now = tm.tv_sec + tm.tv_usec / 1000.0 / 1000.0;
        if (now - report_time > 1) {
            Log::info("bytes transferred = %ld (%.2lf%%)", n_bytes_read, n_bytes_read * 100.0 / (n_writers * g_bytes_per_writer));
            report_time = now;
        }
    }
    xfer_timer.stop();
    Log::info("marshal xfer speed = %.2lf M/s (%d writers, %d bytes per write)",
        n_bytes_read / 1024.0 / 1024.0 / xfer_timer.elapsed(), n_writers, g_chunk_size);
    close(null_fd);

    for (int i = 0; i < n_writers; i++) {
        Pthread_join(th_writers[i], nullptr);
    }
}

static Marshal* marshal_with_size(size_t n) {
    size_t left = n;
    Marshal* m = new Marshal;
    while (left >= 2) {
        *m << "x";
        left -= 2;
    }
    if (left == 1) {
        *m << "";
    }
    verify(m->content_size() == n);
    return m;
}

TEST(marshal, read_from_marshal) {
    Marshal* m1 = marshal_with_size(2);
    const size_t size2 = 19877;
    Marshal* m2 = marshal_with_size(size2);
    EXPECT_EQ(m1->read_from_marshal(*m2, m2->content_size()), size2);
    delete m1;
    delete m2;
}

TEST(marshal_regression, update_tail_when_marshal_is_full_read) {
    const size_t marshal_chunk_size = 8192;
    Marshal* m = marshal_with_size(marshal_chunk_size);
    EXPECT_EQ(m->content_size(), marshal_chunk_size);
    int null_fd = open("/dev/null", O_WRONLY);
    m->write_to_fd(null_fd);
    for (size_t i = 0; i < marshal_chunk_size / 2; i++) {
        *m << "x";   // actually writing 2 bytes (1 for var_int size, 1 for 'x')
    }
    EXPECT_EQ(m->content_size(), marshal_chunk_size);
    m->write_to_fd(null_fd);
    delete m;
}
