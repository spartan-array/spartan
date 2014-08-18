#include <unistd.h>
#include <fcntl.h>
#include <cstdlib>

#include "base/all.h"
#include "rpc/marshal.h"

#include "protobuf/marshal-protobuf.h"
#include "person.pb.h"

using namespace rpc;

TEST(marshal, protobuf) {
    GOOGLE_PROTOBUF_VERIFY_VERSION;

    Person p1, p2;
    Marshal m;
    p1.set_id(1987);
    p1.set_name("Santa Zhang");
    m << p1;
    m >> p2;
    EXPECT_EQ(p2.id(), p1.id());
    EXPECT_EQ(p2.name(), p1.name());
    EXPECT_EQ(p2.id(), 1987);
    EXPECT_EQ(p2.name(), "Santa Zhang");
    EXPECT_EQ(m.content_size(), 0u);

    int n_marshal = 1000000;
    Timer t;
    t.start();
    for (int i = 0; i < n_marshal; i ++) {
        m << p1;
        m >> p2;
    }
    t.stop();
    Log::info("marshal and unmarshal %d person records takes %.2lf seconds, qps=%.0lf",
        n_marshal, t.elapsed(), n_marshal / t.elapsed());

    google::protobuf::ShutdownProtobufLibrary();
}
