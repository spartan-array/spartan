#!/usr/bin/env python

'''Helper module for importing SWIG bindings.

Sets PYTHONPATH when running from the build directory,
and imports all symbols from the SWIG generated code.
''' 

import sys
from os.path import abspath

sys.path += [abspath('../build/.libs'), 
             abspath('../build/python/spartan'), 
             abspath('.')]

from spartan_wrap import *
