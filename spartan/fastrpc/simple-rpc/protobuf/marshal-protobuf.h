#pragma once

#include <sstream>
#include <string>
#include <google/protobuf/message.h>

#include "rpc/marshal.h"

namespace rpc {

inline rpc::Marshal& operator <<(rpc::Marshal& m, const ::google::protobuf::Message& msg) {
    std::ostringstream ostr;
    msg.SerializeToOstream(&ostr);
    m << ostr.str();
    return m;
}

inline rpc::Marshal& operator >>(rpc::Marshal& m, ::google::protobuf::Message& msg) {
    std::string str;
    m >> str;
    std::istringstream istr(str);
    msg.ParseFromIstream(&istr);
    return m;
}

} // namespace rpc
