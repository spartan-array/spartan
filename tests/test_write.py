from spartan import expr
from spartan.util import Assert
from spartan.array import extent
import numpy as np
import test_common
import os

class TestWrite(test_common.ClusterTest):
  def test1(self):
    npa = np.random.random((100, 100))
    np.save('_test_write1', npa)
    np.savez('_test_write2', npa)
    t1 = expr.make_from_numpy('_test_write1.npy')
    t2 = expr.make_from_numpy('_test_write2.npz')
    t3 = expr.make_from_numpy(npa)
    Assert.all_eq(t1.glom(), npa)
    Assert.all_eq(t2.glom(), npa)
    Assert.all_eq(t3.glom(), npa)
    os.system('rm -rf _test_write1.npy _test_write2.npz')

  def test2(self):
    npa = np.random.random(10000)
    np.save('_test_write3', npa)
    np.savez('_test_write4', npa)
    t1 = expr.make_from_numpy('_test_write3.npy')
    t2 = expr.make_from_numpy('_test_write4.npz')
    t3 = expr.make_from_numpy(npa)
    Assert.all_eq(t1.glom(), npa)
    Assert.all_eq(t2.glom(), npa)
    Assert.all_eq(t3.glom(), npa)
    os.system('rm -rf _test_write3.npy _test_write4.npz')

