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

from .arrays import astype, tocoo, size
from .creation import empty, sparse_empty, empty_like
from .creation import zeros, zeros_like, ones, ones_like, eye, identity, full, full_like
from .creation import diagonal, diag, diagflat, sparse_diagonal
from .logic import all, any
from .manipulation import ravel, concatenate
from .mathematics import add, sub, multiply
from .mathematics import power, ln, log, square, sqrt, exp
from .mathematics import abs, maximum, sum, prod
from .srandom import set_random_seed, rand, randn, randint, sparse_rand
from .statistics import max, min, mean, std, bincount, normalize, norm, norm_cdf
from .sorting import argmin, argmax, count_nonzero, count_zero

from .assign import assign
from .retile import retile
from .dot import dot
from .fio import save, load, pickle, unpickle, partial_load, partial_unpickle

from .operator.base import Expr, evaluate, optimized_dag
from .operator.base import eager, lazify, as_array, force, glom
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
