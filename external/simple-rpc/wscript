APPNAME="simple-rpc"
VERSION="0.1"

import os
import sys
from waflib import Logs

if sys.platform == 'darwin' and not os.environ.has_key("CXX"):
    os.environ["CXX"] = "clang++"   # use clang++ as default compiler (for c++11 support on mac)

def options(opt):
    opt.load("compiler_cxx")

def configure(conf):
    conf.load("compiler_cxx")

    if sys.platform == "darwin":
        conf.env.append_value("CXXFLAGS", "-stdlib=libc++")
        conf.env.append_value("LINKFLAGS", "-stdlib=libc++")
    if os.getenv("DEBUG") in ["true", "1"]:
        Logs.pprint("PINK", "debug support enabled")
        conf.env.append_value("CXXFLAGS", "-std=c++11 -Wall -pthread -ggdb".split())
    else:
        conf.env.append_value("CXXFLAGS", "-std=c++11 -Wall -pthread -O3 -ggdb -fno-omit-frame-pointer -DNDEBUG".split())

    conf.env.LIB_PTHREAD = 'pthread'

def build(bld):
    def _depend(target, source, action):
        if source != None and os.path.exists(source) == False:
            Logs.pprint('RED', "'%s' not found!" % source)
            exit(1)
        if os.path.exists(target) == False or os.stat(target).st_mtime < os.stat(source).st_mtime:
            Logs.pprint('PINK', action)
            os.system(action)

    _depend("rpc/rpcgen.py", "rpc/rpcgen.g", "pylib/yapps/main.py rpc/rpcgen.g ; chmod a+x rpc/rpcgen.py")
    _depend("logservice/log_service.h", "logservice/log_service.rpc", "simple-rpc/bin/rpcgen.py logservice/log_service.rpc")
    _depend("test/demo_service.h", "test/demo_service.rpc", "rpc/rpcgen.py test/demo_service.rpc")

    bld.stlib(source=bld.path.ant_glob("rpc/*.cc"), target="simplerpc", includes="rpc", lib="pthread")
    bld.stlib(
        source=bld.path.ant_glob("logservice/*.cc", excl="logservice/log_server.cc"),
        target="logservice",
        includes=". logservice simple-rpc",
        use="PTHREAD")
    bld.stlib(source="test/param_map.cc", includes=".", target = 'test', name = 'test')

    def _prog(source, target, includes=".", use="simplerpc PTHREAD"):
        bld.program(source=source, target=target, includes=includes, use=use)

    _prog("test/demo_client.cc", "demo_client")
    _prog("test/demo_server.cc test/demo_service.cc", "demo_server")
    _prog("test/perftest.cc", "perftest")
    _prog("test/marshal_test.cc", "marshal_test")
    _prog("test/counter_bench.cc", "counter_bench")
    _prog("test/threadpool_bench.cc", "threadpool_bench")

    _prog("logservice/log_server.cc", "log_server", use="logservice simplerpc PTHREAD")
    _prog("test/log_client.cc", "log_client", use="logservice simplerpc PTHREAD")

    # Unit tests and benchmark, inherented from MCP code
    _prog(source="test/param_map_test.cc", target="param_map_test", use="simplerpc test PTHREAD")
    _prog(source="test/callback_test.cc", target="callback_test", use="simplerpc test PTHREAD")
    _prog(source="test/callback_benchmark.cc", target="callback_benchmark", use="simplerpc test PTHREAD")
