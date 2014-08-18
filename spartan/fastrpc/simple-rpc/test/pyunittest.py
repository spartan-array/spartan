#!/usr/bin/env python

import unittest
from unittest import TestCase
import time
import sys
import os
sys.path += os.path.abspath(os.path.join(os.path.split(__file__)[0], "../pylib")),
import simplerpc
from benchmark_service import *
from test_service import *
from threading import Thread

class TestMarshal(TestCase):
    def test_marshal(self):
        m = simplerpc.Marshal()
        assert len(m) == 0
        m.write_v32(45)
        assert len(m) == 1
        assert m.read_v32() == 45
        assert len(m) == 0
        m.write_v64(45)
        assert len(m) == 1
        assert m.read_v64() == 45
        assert len(m) == 0
        m.write_i8(45)
        assert len(m) == 1
        assert m.read_i8() == 45
        assert len(m) == 0
        m.write_i16(45)
        assert len(m) == 2
        assert m.read_i16() == 45
        assert len(m) == 0
        m.write_i32(45)
        assert len(m) == 4
        assert m.read_i32() == 45
        assert len(m) == 0
        m.write_i32(-45)
        assert len(m) == 4
        assert m.read_i32() == -45
        assert len(m) == 0
        m.write_i64(1987)
        assert len(m) == 8
        assert m.read_i64() == 1987
        assert len(m) == 0
        m.write_i64(-1987)
        assert len(m) == 8
        assert m.read_i64() == -1987
        assert len(m) == 0
        m.write_double(-1.987)
        assert len(m) == 8
        assert m.read_double() == -1.987
        assert len(m) == 0
        m.write_str("hello world!")
        print len(m)
        print m.read_str()
        print len(m)
        p = point3(x=3.0, y=4.0, z=5.0)
        m.write_obj(p, "point3")
        print len(m)
        print m.read_obj("point3")
        print len(m)
        comp_s = set(["abc", "def"])
        comp_d = {}
        comp_d[("1", "2")] = []
        comp_d[("a", "b")] = [[("e", "f"), ("g", "hi")], []]
        comp = complex_struct(d=comp_d, s=comp_s, e=empty_struct())
        m.write_obj(comp, "complex_struct")
        print m.read_obj("complex_struct")

class TestUtils(TestCase):
    def test_marshal_wrap(self):
        from simplerpc.server import MarshalWrap

        def a_add_b(a, b):
            return a + b

        f = MarshalWrap(a_add_b, ["rpc::i32", "rpc::i32"], ["rpc::i32"])

        req_m = simplerpc.Marshal()
        req_m.write_i32(3)
        req_m.write_i32(4)

        rep_m_id = f(req_m.id)
        rep_m = simplerpc.Marshal(id=rep_m_id)

        assert len(rep_m) == 4
        assert rep_m.read_i32() == 7
        assert len(rep_m) == 0

        def a_swap_b(a, b):
            return b, a

        f = MarshalWrap(a_swap_b, ["rpc::i32", "rpc::i32"], ["rpc::i32", "rpc::i32"])

        req_m = simplerpc.Marshal()
        req_m.write_i32(3)
        req_m.write_i32(4)

        rep_m_id = f(req_m.id)
        rep_m = simplerpc.Marshal(id=rep_m_id)

        assert len(rep_m) == 8
        assert rep_m.read_i32() == 4
        assert rep_m.read_i32() == 3
        assert len(rep_m) == 0

        def hello_world():
            print "** hello world! **"

        f = MarshalWrap(hello_world, [], [])

        req_m = simplerpc.Marshal()
        assert f(req_m.id) == 0 # NULL rpc return

        def hello_greetings(greetings):
            print "** hello %s **" % greetings

        f = MarshalWrap(hello_greetings, ["std::string"], [])

        req_m = simplerpc.Marshal()
        req_m.write_str("simple-rpc")
        assert f(req_m.id) == 0 # NULL rpc return

class TestAsync(TestCase):
    def test_async_rpc(self):
        s = simplerpc.Server()
        class LazyMath(MathService):
            def gcd(self, a, b):
                print "server zzz..."
                time.sleep(1)
                print "server wake up"
                while True:
                    r = a % b
                    if r == 0:
                        return b
                    else:
                        a = b
                        b = r
        s.reg_svc(LazyMath())
        s.start("0.0.0.0:8848")
        c = simplerpc.Client()
        c.connect("127.0.0.1:8848")
        mp = MathProxy(c)
        n_jobs = 10
        fu_all = []
        for i in range(n_jobs):
            print "client calling..."
            fu = mp.async_gcd(124, 84)
            fu_all += fu,
        for fu in fu_all:
            print "client waiting..."
            print "error code: %d" % fu.error_code
            print "client got result:", fu.result

class TestMultithread(TestCase):
    def test_mt_rpc(self):
        s = simplerpc.Server()
        class MyMath(MathService):
            def gcd(self, a, b):
                while True:
                    r = a % b
                    if r == 0:
                        return b
                    else:
                        a = b
                        b = r
        s.reg_svc(MyMath())
        s.start("0.0.0.0:8848")
        n_th = 10
        n_jobs = 200
        start = time.time()
        class MyThread(Thread):
            def run(self):
                c = simplerpc.Client()
                c.connect("127.0.0.1:8848")
                mp = MathProxy(c)
                for i in range(n_jobs):
                    mp.sync_gcd(124, 84)
        th = []
        for i in range(n_th):
            t = MyThread()
            t.start()
            th += t,
        for t in th:
            t.join()
        end = time.time()
        print "mt: qps = %.2lf" % (1.0 * n_th * n_jobs / (end - start))

class TestRpcGen(TestCase):
    def test_struct_gen(self):
        p = point3(x=3.0, y=4.0, z=5.0)
        print p


    def test_timedwait(self):
        class BS(BenchmarkService):
            def sleep(self, sec):
                time.sleep(sec)
        s = simplerpc.Server()
        s.reg_svc(BS())
        s.start("0.0.0.0:8848")
        c = simplerpc.Client()
        c.connect("127.0.0.1:8848")
        bp = BenchmarkProxy(c)
        fu = bp.async_sleep(2.3)
        fu.wait(1.0)
        print "done"

    def test_service_gen(self):
        s = simplerpc.Server()
        class MyMath(MathService):
            def gcd(self, a, b):
                while True:
                    r = a % b
                    if r == 0:
                        return b
                    else:
                        a = b
                        b = r
        s.reg_svc(MyMath())
        s.start("0.0.0.0:8848")

        c = simplerpc.Client()
        c.connect("127.0.0.1:8848")

        # raw marshal handling
        print c.sync_call(MathService.GCD, [124, 84], ["rpc::i64", "rpc::i64"], ["rpc::i64"])

        mp = MathProxy(c)
        print mp.sync_gcd(124, 84)
        print "begin 10000 sync_gcd operation"
        start = time.time()
        for i in range(10000):
            mp.sync_gcd(124, 84)
        print "done 10000 sync_gcd operation"
        end = time.time()
        print "qps = %.2lf" % (10000.0 / (end - start))

        n_async = 40000
        print "begin 100000 async_gcd operation"
        start = time.time()
        fu_list = []
        for i in range(n_async):
            fu_list += mp.async_gcd(124, 84),
        print "now waiting..."
        for fu in fu_list:
            fu.wait()
        print "done 100000 async_gcd operation"
        end = time.time()
        print "qps = %.2lf" % (n_async * 1.0 / (end - start))

        c.close()

if __name__ == "__main__":
    unittest.main()
