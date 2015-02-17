import math
import sys
import unittest

import spartan
import numpy as np
from spartan import expr, util
from spartan.array import distarray, extent
from spartan.util import Assert
import test_common

TEST_SIZE = 10


def add_one_extent(v, ex):
  result = v.fetch(ex) + 1
  util.log_info('AddOne: %s, %s', ex, result)
  yield (ex, result)


def add_one_tile(tile):
  return tile + 1


class SliceTest(test_common.ClusterTest):
  def test_slice_get(self):
    x = expr.arange((TEST_SIZE, TEST_SIZE))
    z = x[5:8, 5:8]
    val = z.evaluate()
    nx = np.arange(TEST_SIZE*TEST_SIZE).reshape(TEST_SIZE, TEST_SIZE)
    Assert.all_eq(val.glom(), nx[5:8, 5:8])

  def test_slice_map(self):
    x = expr.arange((TEST_SIZE, TEST_SIZE))
    z = x[5:8, 5:8]
    z = expr.map(z, add_one_tile)
    print z
    nx = np.arange(TEST_SIZE*TEST_SIZE).reshape(TEST_SIZE, TEST_SIZE)

    Assert.all_eq(z.glom(), nx[5:8, 5:8] + 1)

  def test_slice_shuffle(self):
    x = expr.arange((TEST_SIZE, TEST_SIZE))
    z = x[5:8, 5:8]
    z = expr.shuffle(z, add_one_extent)
    val = z.evaluate()
    nx = np.arange(TEST_SIZE*TEST_SIZE).reshape(TEST_SIZE, TEST_SIZE)

    Assert.all_eq(val.glom(), nx[5:8, 5:8] + 1)

  def test_slice_map2(self):
    x = expr.arange((10, 10, 10), dtype=np.int)
    nx = np.arange(10 * 10 * 10, dtype=np.int).reshape((10, 10, 10))

    y = x[:, :, 0]
    z = expr.map(y, lambda tile: tile + 13)
    val = z.glom()

    Assert.all_eq(val.reshape(10, 10), nx[:, :, 0] + 13)

  def test_from_slice(self):
    print extent.from_slice((slice(None), slice(None), 0), [100, 100, 100])

  def test_slice_reduce(self):
    x = expr.arange((TEST_SIZE, TEST_SIZE, TEST_SIZE), dtype=np.int)
    nx = np.arange(TEST_SIZE * TEST_SIZE * TEST_SIZE, dtype=np.int).reshape((TEST_SIZE, TEST_SIZE, TEST_SIZE))
    y = x[:, :, 0].sum()
    val = y.glom()

    Assert.all_eq(val, nx[:, :, 0].sum())

  def test_slice_sub(self):
    a = expr.arange((TEST_SIZE,), dtype=np.int)
    v = (a[1:] - a[:-1])
    print expr.optimize(v)
    v = v.glom()
    print v

    na = np.arange(TEST_SIZE, dtype=np.int)
    nv = na[1:] - na[:-1]
    Assert.all_eq(v, nv)

if __name__ == '__main__':
  rest = spartan.config.initialize(sys.argv)
  unittest.main(argv=rest)
