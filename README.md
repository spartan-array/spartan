## Spartan

Spartan is a distributed array engine, built on top of a Piccolo-style
key-value store.

### Installation

pip install [--user] spartan

### Usage

Operations in Spartan look superficially like numpy array operations, but
actually are composed into a deferred expression tree.  For example:

    >> In [3]: x = spartan.ones((10, 10))
    >> In [4]: x

    MapExpr {
      local_dag = None,
      fn_kw = LazyDict {
        vals = {}
      },
      children = LazyList {
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

### Running

python benchmarks/benchmark_lreg.py

