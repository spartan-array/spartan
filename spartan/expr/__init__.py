#!/usr/bin/env python

"""
Lazy expression DAGs.

All expression operations are subclasses of the `Expr` class, which
by default performs all operations lazily.

Operations are built up using a few high-level operations -- these all
live in their own modules:

* Create a new distributed array `spartan.expr.ndarray`
* Map over an array :py:mod:`spartan.expr.map` and `spartan.expr.shuffle`
* Reduce over an array `spartan.expr.reduce`
* Apply a stencil/convolution to an array `spartan.expr.stencil`
* Slicing/indexing `spartan.expr.index`.   

Optimizations on DAGs live in `spartan.expr.optimize`.

"""

from base import Expr, evaluate, dag, glom, eager, lazify, force, make_primitive, \
  NotShapeable
import numpy as np
from spartan import util

from ..array import extent
from ..array.extent import index_for_reduction, shapes_match
from ..util import Assert
from .builtins import *
from .map import map
from .ndarray import ndarray
from .outer import outer
from .reduce import reduce
from .shuffle import shuffle


Expr.outer = outer
Expr.sum = sum
Expr.mean = mean
Expr.astype = astype
Expr.ravel = ravel
Expr.argmin = argmin
Expr.argmax = argmax
