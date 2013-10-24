"""
Spartan: A distributed array language.

"""

from cluster import start_cluster
from wrap import *
from expr import *

# force configuration settings to load.
import spartan.expr.optimize as _