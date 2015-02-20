#!/usr/bin/env python

"""
Definitions of expressions and optimizations.

In Spartan, operations are not performed immediately.  Instead, they are
represented using a graph of `Expr` nodes.  Expression graphs can be
evaluated using the `Expr.evaluate` methods.

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

from .arrays import astype, tocoo, size
from .creation import empty, sparse_empty, empty_like
from .creation import zeros, zeros_like, ones, ones_like, eye, identity, full, full_like
from .creation import arange, diagonal, diag, diagflat, sparse_diagonal
from .logic import all, any, equal, not_equal, greater, greater_equal, less, less_equal
from .logic import logical_and, logical_or, logical_xor
from .manipulation import ravel, concatenate
from .mathematics import add, sub, multiply, divide, true_divide, floor_divide
from .mathematics import reciprocal, negative, fmod, mod, remainder
from .mathematics import power, ln, log, square, sqrt, exp
from .mathematics import abs, maximum, minimum, sum, prod
from .srandom import set_random_seed, rand, randn, randint, sparse_rand
from .statistics import max, min, mean, std, bincount, normalize, norm, norm_cdf
from .sorting import argmin, argmax, count_nonzero, count_zero

from .assign import assign
from .retile import retile
from .dot import dot
from .fio import save, load, pickle, unpickle, partial_load, partial_unpickle

from .operator.base import Expr, evaluate, optimized_dag
from .operator.base import eager, lazify, as_array, glom
from .operator.base import NotShapeable, newaxis
from .operator.broadcast import broadcast
from .operator.checkpoint import checkpoint
from .operator.map import map, map2
from .operator.map_with_location import map_with_location
from .operator.ndarray import ndarray
from .operator.outer import outer
from .operator.optimize import optimize
from .operator.region_map import region_map
from .operator.reshape import reshape
from .operator.reduce import reduce
from .operator.sort import sort, argsort, argpartition, partition
from .operator.shuffle import shuffle
from .operator.scan import scan
from .operator.stencil import stencil, maxpool, _convolve
from .operator.tile_operation import tile_operation
from .operator.transpose import transpose
from .operator.write_array import write, from_numpy, from_file, from_file_parallel


Expr.all = all
Expr.any = any
Expr.argmax = argmax
Expr.argmin = argmin
Expr.argpartition = argpartition
Expr.argsort = argsort
Expr.astype = astype
Expr.diagonal = diagonal
Expr.dot = dot
Expr.fill = full_like
Expr.flat = None
Expr.flatten = ravel
Expr.outer = None
Expr.max = max
Expr.mean = mean
Expr.min = min
Expr.ndim = None
Expr.nonzero = None
Expr.partition = partition
Expr.prod = prod
Expr.ravel = ravel
Expr.reshape = reshape
Expr.std = std
Expr.sum = sum
Expr.transpose = transpose
Expr.T = property(transpose)

from ..array import distarray
import mathematics

distarray.DistArray.evaluate = evaluate
distarray.DistArray.__add__ = add
distarray.DistArray.__sub__ = sub
distarray.DistArray.__mul__ = multiply
distarray.DistArray.__mod__ = mod
distarray.DistArray.__div__ = divide
distarray.DistArray.__eq__ = equal
distarray.DistArray.__ne__ = not_equal
distarray.DistArray.__lt__ = less
distarray.DistArray.__gt__ = greater
distarray.DistArray.__and__ = logical_and
distarray.DistArray.__or__ = logical_or
distarray.DistArray.__xor = logical_xor
distarray.DistArray.__pow__ = power
distarray.DistArray.__neg__ = negative
distarray.DistArray.__rsub__ = mathematics._rsub
distarray.DistArray.__radd__ = add
distarray.DistArray.__rmul__ = multiply
distarray.DistArray.__rdiv__ = mathematics._rdivide
distarray.DistArray.all = all
distarray.DistArray.any = any
distarray.DistArray.argmax = argmax
distarray.DistArray.argmin = argmin
distarray.DistArray.argpartition = argpartition
distarray.DistArray.argsort = argsort
distarray.DistArray.astype = astype
distarray.DistArray.diagonal = diagonal
distarray.DistArray.dot = dot
distarray.DistArray.fill = full_like
distarray.DistArray.flat = None
distarray.DistArray.flatten = ravel
distarray.DistArray.outer = None
distarray.DistArray.max = max
distarray.DistArray.mean = mean
distarray.DistArray.min = min
distarray.DistArray.ndim = None
distarray.DistArray.nonzero = None
distarray.DistArray.partition = partition
distarray.DistArray.prod = prod
distarray.DistArray.ravel = ravel
distarray.DistArray.reshape = reshape
distarray.DistArray.std = std
distarray.DistArray.sum = sum
distarray.DistArray.transpose = transpose
distarray.DistArray.T = property(transpose)
