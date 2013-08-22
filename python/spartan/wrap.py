#!/usr/bin/env python

'''Helper module for importing SWIG bindings.

Sets PYTHONPATH when running from the build directory,
and imports all symbols from the SWIG generated code.
''' 

import sys
sys.path += ['../build/.libs', '../build/python/spartan/']

from spartan_wrap import *