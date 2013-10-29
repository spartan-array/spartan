#!/usr/bin/env python

"""
Lazy expression DAGs.

All expression operations are subclasses of the `Expr` class, which
by default performs all operations lazily.

Operations are built up using a few high-level operations -- these all
live in their own modules:

* Create a new distributed array `spartan.expr.ndarray`
* Map over an array :py:mod:`spartan.expr.map_tiles` and `spartan.expr.map_extents`
* Reduce over an array `spartan.expr.reduce_extents`
* Apply a stencil/convolution to an array `spartan.expr.stencil`
* Slicing/indexing `spartan.expr.index`.   

Optimizations on DAGs live in `spartan.expr.optimize`.

"""

from ..dense import extent
from ..dense.extent import index_for_reduction, shapes_match
from ..util import Assert
from .map_extents import map_extents
from .map_tiles import map_tiles
from .ndarray import ndarray
from .outer import outer
from .reduce_extents import reduce_extents
from base import Expr, evaluate, dag, glom, eager, lazify, force, make_primitive, NotShapeable
from spartan import util
import numpy as np

from .builtins import *


Expr.outer = outer
Expr.sum = sum
Expr.mean = mean
Expr.astype = astype
Expr.ravel = ravel
Expr.argmin = argmin
