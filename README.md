# Spartan

Spartan is a library for distributed array programming.  Programmers
build up array expressions (using Numpy-like operations).  These 
expressions are then compiled and optimized and run on a distributed
array backend across multiple machines.

## Installation

#### From PyPi (not necessarily up-to-date)
    
    pip install [--user] spartan

#### From source

    pip install --user cython
    git clone http://github.com/rjpower/spartan.git
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

    nosetests tests/

There are a few benchmarks to test performance as well:

    python benchmarks/benchmark_*.py

