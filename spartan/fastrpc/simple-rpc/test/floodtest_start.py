#!/usr/bin/env python

import os
import sys
sys.path += os.path.abspath(os.path.join(os.path.split(__file__)[0], "../pylib")),
import simplerpc
from floodtest_service import *

fpath = os.path.join(os.path.split(__file__)[0], "floodtest-servers.txt")
with open(fpath) as f:
    nodelist = []
    for l in f:
        l = l.strip()
        if l.startswith("#") or l == "":
            continue
        nodelist += l,
    for n in nodelist:
        c = simplerpc.Client()
        print "connect to %s" % n
        c.connect(n)
        proxy = FloodProxy(c)
        proxy.sync_update_node_list(nodelist)
