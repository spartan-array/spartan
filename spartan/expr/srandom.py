'''
Basic numpy style random operations on arrays.

These include --
*
'''
import sys
import __builtin__
import numpy as np
import scipy.sparse as sp

from .operator.map import map, map2
from .operator.map_with_location import map_with_location
from .operator.reduce import reduce
from .operator.ndarray import ndarray
from .operator.optimize import disable_parakeet, not_idempotent
from .. import util, blob_ctx
from ..array import extent
from ..array.extent import index_for_reduction, shapes_match
from ..util import Assert


@disable_parakeet
def _set_random_seed_mapper(input):
  import time
  import random
  import os
  np.random.seed((int(time.time() * 100000) + random.randint(0, 1000000) +
                  os.getpid()) % 4294967295)
  return np.zeros((1, ))


def set_random_seed():
  ctx = blob_ctx.get()
  map(ndarray((ctx.num_workers, ), dtype=np.int32,
              tile_hint=(1, )), fn=_set_random_seed_mapper).evaluate()


@disable_parakeet
def _make_rand(input):
  return np.random.rand(*input.shape)


@disable_parakeet
def _make_randn(input):
  return np.random.randn(*input.shape)


@disable_parakeet
def _make_randint(input, low=0, high=10):
  return np.random.randint(low, high, size=input.shape)


@disable_parakeet
def _make_sparse_rand(input,
                      density=None,
                      dtype=None,
                      format='csr'):
  Assert.eq(len(input.shape), 2)

  return sp.rand(input.shape[0],
                 input.shape[1],
                 density=density,
                 format=format,
                 dtype=dtype)


@not_idempotent
def rand(*shape, **kw):
  '''
  Return a random array sampled from the uniform distribution on [0, 1).

  :param tile_hint: A tuple indicating the desired tile shape for this array.
  '''
  tile_hint = None
  if 'tile_hint' in kw:
    tile_hint = kw['tile_hint']
    del kw['tile_hint']

  assert len(kw) == 0, 'Unknown keywords %s' % kw

  for s in shape: assert isinstance(s, (int, long))
  return map(ndarray(shape, dtype=np.float, tile_hint=tile_hint),
             fn=_make_rand)


@not_idempotent
def randn(*shape, **kw):
  '''
  Return a random array sampled from the standard normal distribution.

  :param tile_hint: A tuple indicating the desired tile shape for this array.
  '''
  tile_hint = None
  if 'tile_hint' in kw:
    tile_hint = kw['tile_hint']
    del kw['tile_hint']

  for s in shape: assert isinstance(s, (int, long))
  return map(ndarray(shape, dtype=np.float, tile_hint=tile_hint), fn=_make_randn)


@not_idempotent
def randint(*shape, **kw):
  '''
  Return a random integer array from the "discrete uniform" distribution in the interval [`low`, `high`).

  :param low: Lowest (signed) integer to be drawn from the distribution.
  :param high: Largest (signed) integer to be drawn from the distribution.
  :param tile_hint: A tuple indicating the desired tile shape for this array.
  '''
  tile_hint = None
  if 'tile_hint' in kw:
    tile_hint = kw['tile_hint']
    del kw['tile_hint']

  for s in shape: assert isinstance(s, (int, long))
  return map(ndarray(shape, dtype=np.float, tile_hint=tile_hint), fn=_make_randint, fn_kw=kw)


@not_idempotent
def sparse_rand(shape,
                density=0.001,
                format='lil',
                dtype=np.float32,
                tile_hint=None):
  '''Make a distributed sparse random array.

  Random values are chosen from the uniform distribution on [0, 1).

  Args:
    density(float): Fraction of values to be filled
    format(string): Sparse tile format (lil, coo, csr, csc).
    dtype(np.dtype): Datatype of array.
    tile_hint(tuple or None): Shape of array tiles.

  Returns:
    Expr:
  '''

  for s in shape: assert isinstance(s, (int, long))
  return map(ndarray(shape, dtype=dtype, tile_hint=tile_hint, sparse=True),
             fn=_make_sparse_rand,
             fn_kw={'dtype': dtype,
                    'density': density,
                    'format': format})
