#!/usr/bin/env python

import sys
import os
import argparse
import time
sys.path += os.path.abspath(os.path.join(os.path.split(__file__)[0], "../pylib")),
import simplerpc
from benchmark_service import *

def main():
    argparser = argparse.ArgumentParser(prog=sys.argv[0])
    argparser.add_argument("-c", dest="client_addr", help="server address (ip:port)")
    argparser.add_argument("-s", dest="server_addr", help="client address (ip:port)")
    opt = argparser.parse_args(sys.argv[1:])
    if opt.client_addr:
        c = simplerpc.Client()
        c.connect(opt.client_addr)
        bp = BenchmarkProxy(c)
        counter = 0
        last_time = time.time()
        while True:
            now = time.time()
            if now - last_time > 1.0:
                print "qps=%d" % (counter / (now - last_time))
                last_time = now
                counter = 0
            bp.sync_nop("")
            counter += 1

    elif opt.server_addr:
        s = simplerpc.Server()

        class MyBenchmarkService(BenchmarkService):
            def nop(self, in0):
                pass

        s.reg_svc(MyBenchmarkService())

        s.start(opt.server_addr)
        while True:
            time.sleep(1)

    else:
        argparser.print_help()

if __name__ == "__main__":
    main()
