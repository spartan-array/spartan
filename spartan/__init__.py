"""
Spartan: A distributed array language.
"""

try:
  import pyximport
  pyximport.install()
except:
  pass


from cluster import start_cluster
from expr import *