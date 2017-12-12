# Spartan

[![Build Status](https://travis-ci.org/spartan-array/spartan.svg?branch=master)](https://travis-ci.org/spartan-array/spartan)

Spartan is a library for distributed array programming.  Programmers
build up array expressions (using Numpy-like operations).  These 
expressions are then compiled and optimized and run on a distributed
array backend across multiple machines.

Check out the
[tutorial on the wiki.](https://github.com/spartan-array/spartan/wiki/Tutorial)
## Publication
[Spartan: A Distributed Array Framework with Smart Tiling, USENIX ATC'15](https://www.usenix.org/system/files/conference/atc15/atc15-paper-huang-chien-chin.pdf)

## Installation

#### From PyPI (not necessarily up-to-date)
    
    pip install [--user] spartan

#### From source

    # For numpy and scipy, we suggest you use binary install to
    # get better performance.
    apt-get install python-numpy python-scipy libzmq3-dev
    pip install --user dsltools
    pip install --user pyzmq
    pip install --user cython
    pip install --user parakeet
    pip install --user scikit-learn
    pip install --user traits
    git clone https://github.com/spartan-array/spartan.git
    cd spartan
    python setup.py develop --user

## Usage

Operations in Spartan look superficially like numpy array operations, but
actually are composed into a deferred expression tree.  For example:

    >> In [3]: x = spartan.ones((10, 10))
    >> In [4]: x

    MapExpr {
      local_dag = None,
      fn_kw = DictExpr {
        vals = {}
      },
      children = ListExpr {
        vals = [
        [0] = NdArrayExpr {
          combine_fn = None,
          dtype = <type 'float'>,
          _shape = (10, 10),
          tile_hint = None,
          reduce_fn = None
        }
        ]
      },
      map_fn = <function <lambda> at 0x3dbae60>
    }


Expressions are combined together lazily until they are *forced* -- this
is caused by a call to the ``force`` method.

## Running

Tests can be run using nosetests `pip install --user nose`.

    pip install --user nose
    nosetests tests/

There are a few benchmarks for performance testing, see

    tests/benchmark_*.py

