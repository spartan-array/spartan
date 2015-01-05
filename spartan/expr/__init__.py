#!/usr/bin/env python

"""
Definitions of expressions and optimizations.

In Spartan, operations are not performed immediately.  Instead, they are
represented using a graph of `Expr` nodes.  Expression graphs can be
evaluated using the `Expr.evaluate` or `Expr.force` methods.

The `base` module contains the definition of `Expr`, the base class for all
types of expressions.  It also defines subclasses for wrapping common
Python values: lists (`ListExpr`), dicts (`DictExpr`) and tuples ((`TupleExpr`).

Operations are built up using a few high-level operations -- these all
live in their own modules:

* Create a new distributed array `spartan.expr.ndarray`
* Map over an array :py:mod:`spartan.expr.map` and `spartan.expr.shuffle`
* Reduce over an array `spartan.expr.reduce`
* Apply a stencil/convolution to an array `spartan.expr.stencil`
* Slicing/indexing `spartan.expr.index`.

Optimizations on DAGs live in `spartan.expr.optimize`.
"""

from base import Expr, evaluate, optimized_dag, glom, eager, lazify, as_array, force, NotShapeable, newaxis
from .builtins import *
from .assign import assign
from .map import map
from .map_with_location import map_with_location
from .region_map import region_map
from .tile_operation import tile_operation
from .ndarray import ndarray
from .outer import outer
from .reduce import reduce
from .shuffle import shuffle
from .scan import scan
from .write_array import write, from_numpy, from_file, from_file_parallel
from .checkpoint import checkpoint
from .fio import save, load, pickle, unpickle, partial_load, partial_unpickle
from .reshape import reshape
from .retile import retile
from .transpose import transpose
from .dot import dot
from .sort import sort, argsort, argpartition

Expr.outer = outer
Expr.sum = sum
Expr.mean = mean
Expr.astype = astype
Expr.ravel = ravel
Expr.argmin = argmin
Expr.argmax = argmax
