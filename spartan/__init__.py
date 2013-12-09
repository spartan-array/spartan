"""
Spartan: A distributed array language.
"""

try:
  #import pyximport
  #pyximport.install()
  pass
except:
  print 'Pyximport failed (this is likely not a problem unless you are changing Cython files)'


from cluster import start_cluster
from expr import *
