"""
Spartan: A distributed array language.
"""

try:
  #import pyximport
  #pyximport.install()
  pass
except:
  print 'Pyximport failed (this is likely not a problem unless you are changing Cython files)'


import sys

from .expr import *
from . import config
from .config import FLAGS
from .cluster import start_cluster


CTX = None
def initialize(argv=None):
  global CTX
  
  if CTX is not None:
    return CTX
  
  if argv is None: argv = sys.argv
  config.parse(argv)
  CTX = start_cluster(FLAGS.num_workers, FLAGS.cluster)
  return CTX

def shutdown():
  global CTX
  CTX.shutdown()
  CTX = None