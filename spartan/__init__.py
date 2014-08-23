"""
Spartan: A distributed array language.

Spartan expressions and optimizations are defined in the `spartan.expr` package.
The RPC and serialization library are defined in `spartan.rpc`.  

A Spartan execution environment consists of a master process and one or more 
workers; these are defined in the `spartan.master` and `spartan.worker` modules
respectively.

Workers communicate with each other and the master via RPC; the RPC protocol is 
based on ZeroMQ and is located in the `spartan.rpc` package.  RPC messages used
in Spartan are defined in `spartan.core`.

For convenience, all array operations are routed through a "context"; this 
tracks ownership (which worker stores each part of an array) and simplifies
sending out RPC messages to many workers at once.  This context, is for historical
reasons located in `spartan.blob_ctx`.
"""

try:
  #import pyximport
  #pyximport.install()
  pass
except:
  print 'Pyximport failed (this is likely not a problem unless you are changing Cython files)'


import signal
import sys

import core
from .expr import *
from . import config
from .config import FLAGS
from .cluster import start_cluster


CTX = None


def signal_shutdown(signum, frame):
  '''Shutdown Spartan and revert to orignal signal handler.'''
  shutdown()
  if signum == signal.SIGINT:
    signal.signal(signum, signal.default_int_handler)  # KeyboardInterrupt
  else:
    signal.signal(signum, signal.SIG_DFL)  # default signal handler


def initialize(argv=None):
  global CTX

  if CTX is not None:
    return CTX

  if argv is None:
    argv = sys.argv

  config.parse(argv)
  CTX = start_cluster(FLAGS.num_workers, FLAGS.cluster)

  # If spartan is running in cluster mode, we need to shutdown the workers
  #   manually. The user can either explicitly call `spartan.shutdown()` or
  #   they can use the interrupt signal (signal.SIGINT) a.k.a. `ctrl-C`
  if FLAGS.cluster:
    signal.signal(signal.SIGINT, signal_shutdown)
  return CTX


def shutdown():
  global CTX
  CTX.shutdown()
  CTX = None
