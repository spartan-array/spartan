#include "rpc/marshal.h"
#include "rpc/utils.h"

using namespace rpc;

int main() {
    Marshal m;
    rpc::i32 a = 4;
    Log::debug("content size = %d", m.content_size());
    m << a;
    Log::debug("content size = %d", m.content_size());
    rpc::i32 b = 9;
    m >> b;
    Log::debug("content size = %d", m.content_size());
    verify(a == b);
    return 0;
}

