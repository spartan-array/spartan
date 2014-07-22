#!/usr/bin/env python

import spartan
import numpy as np
import test_common
from spartan.util import Assert

class BuiltinTest(test_common.ClusterTest):
  def test_std(self):
    # 1d array.
    np_1d = np.array([1, 2, 3, 4])
    sp_1d = spartan.from_numpy(np_1d)
    Assert.float_close(spartan.std(sp_1d).glom(), np.std(np_1d))

    # 2d array with auto-flattening.
    np_2d = np_1d.reshape((2, 2))
    sp_2d = sp_1d.reshape((2, 2))
    Assert.float_close(spartan.std(sp_2d).glom(), np.std(np_2d))

    # With axis parameter.
    Assert.all_close(spartan.std(sp_2d, 0).glom(), np.std(np_2d, 0))
    Assert.all_close(spartan.std(sp_2d, 1).glom(), np.std(np_2d, 1))

    np_big = np.arange(256).reshape((16, 16))
    sp_big = spartan.from_numpy(np_big)
    Assert.all_close(spartan.std(sp_big, 0).glom(), np.std(np_big, 0))
    Assert.all_close(spartan.std(sp_big, 1).glom(), np.std(np_big, 1))


if __name__ == '__main__':
  test_common.run(__file__)
