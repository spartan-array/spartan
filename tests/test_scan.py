#!/usr/bin/env python
'''Test the scan higher level operator.

``np.cumsum(...)`` automatically flattens the matrix while spartan does not.
This is why the result of ``np.cumsum(...)`` is reshaped.

'''
import numpy as np
from scipy import sparse
import test_common

from spartan import from_numpy, scan, sparse_diagonal
from spartan.util import Assert

ARRAY_SIZE = (10, 10)
tile_hint = [5, 5]

class ScanTest(test_common.ClusterTest):
  @staticmethod
  def all_eq_helper(spa, npa, axis=None):
    Assert.all_eq(scan(spa, axis=axis).glom(),
                  np.cumsum(npa, axis).reshape(ARRAY_SIZE))


  def test_sum_scan(self):
    source = np.ones(ARRAY_SIZE, np.float32)
    self.all_eq_helper(from_numpy(source, tile_hint), source)
    for axis in range(len(ARRAY_SIZE)):
      self.all_eq_helper(from_numpy(source, tile_hint), source, axis)


  def test_sparse_scan(self):
    # np.cumsum does not support sparse matrices, so they must be converted
    #   to dense matrices first.
    np_sparse = sparse.eye(*ARRAY_SIZE).todense()
    sp_sparse = sparse_diagonal(ARRAY_SIZE, np.float32, tile_hint)
    Assert.all_eq(
        scan(sp_sparse, reduce_fn=lambda x, **kw:x.sum(axis=kw['axis']),
                        scan_fn=lambda x, **kw: x.cumsum(axis=kw['axis']),
                        axis=None).glom(),
        np.cumsum(np_sparse).reshape(ARRAY_SIZE))

    for axis in range(len(ARRAY_SIZE)):
      Assert.all_eq(
          scan(sp_sparse, reduce_fn=lambda x, **kw:x.sum(axis=kw['axis']),
                          scan_fn=lambda x, **kw: x.cumsum(axis=kw['axis']),
                          axis=axis).glom(),
          np.cumsum(np_sparse, axis).reshape(ARRAY_SIZE))


if __name__ == '__main__':
  test_common.run(__file__)
