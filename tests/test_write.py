from spartan import expr
from spartan.util import Assert
from spartan.array import extent
import numpy as np
import test_common
import os

class TestWrite(test_common.ClusterTest):
  def test_from_np1d(self):
    npa = np.random.random((100, 100))
    np.save('_test_write1', npa)
    np.savez('_test_write2', npa)
    t1 = expr.from_file('_test_write1.npy', sparse = False)
    t2 = expr.from_file('_test_write2.npz', sparse = False)
    t3 = expr.from_numpy(npa)
    Assert.all_eq(t1.glom(), npa)
    Assert.all_eq(t2.glom(), npa)
    Assert.all_eq(t3.glom(), npa)
    os.system('rm -rf _test_write1.npy _test_write2.npz')

  def test_from_np2d(self):
    npa = np.random.random(10000)
    np.save('_test_write3', npa)
    np.savez('_test_write4', npa)
    t1 = expr.from_file('_test_write3.npy', sparse = False)
    t2 = expr.from_file('_test_write4.npz', sparse = False)
    t3 = expr.from_numpy(npa)
    Assert.all_eq(t1.glom(), npa)
    Assert.all_eq(t2.glom(), npa)
    Assert.all_eq(t3.glom(), npa)
    os.system('rm -rf _test_write3.npy _test_write4.npz')

  def test_slices_from_np(self):
    npa = np.random.random(10000).reshape(100, 100)
    slices1 = (slice(0, 50), slice(0, 50))
    slices2 = (slice(0, 50), slice(50, 100))
    slices3 = (slice(50, 100), slice(0, 50))
    slices4 = (slice(50, 100), slice(50, 100))
    t1 = expr.randn(100, 100)
    t2 = expr.write(t1, slices1, npa, slices1)
    t3 = expr.write(t2, slices2, npa, slices2)
    t4 = expr.write(t3, slices3, npa, slices3)
    t5 = expr.write(t4, slices4, npa, slices4)
    Assert.all_eq(t5.glom(), npa)

  def test_slices_from_slices(self):
    slices1 = (slice(0, 50), slice(0, 50))
    slices2 = (slice(0, 50), slice(50, 100))
    slices3 = (slice(50, 100), slice(0, 50))
    slices4 = (slice(50, 100), slice(50, 100))
    t1 = expr.randn(100, 100)
    t2 = expr.randn(100, 100)
    t3 = expr.write(t2, slices1, t1, slices1)
    t4 = expr.write(t3, slices2, t1, slices2)
    t5 = expr.write(t4, slices3, t1, slices3)
    t6 = expr.write(t5, slices4, t1, slices4)
    Assert.all_eq(t1.glom(), t6.glom())

    dst = np.arange(0, 2500).reshape(50, 50)
    t11 = expr.write(t1, slices1, dst, slices1)
    t12 = expr.write(t11, slices2, dst, slices1)
    t13 = expr.write(t12, slices3, dst, slices1)
    t14 = expr.write(t13, slices4, dst, slices1)

    tmp = expr.write(expr.randn(100, 100), slices4, dst, slices1)
    t21 = expr.write(t2, slices1, tmp, slices4)
    t22 = expr.write(t21, slices2, tmp, slices4)
    t23 = expr.write(t22, slices3, tmp, slices4)
    t24 = expr.write(t23, slices4, tmp, slices4)
    Assert.all_eq(t14.glom(), t24.glom())

