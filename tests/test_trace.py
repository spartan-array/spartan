from spartan import expr
from spartan.util import Assert
from spartan import util
from spartan.array import extent
import numpy as np
from scipy import sparse as sp
import test_common

class TestTrace(test_common.ClusterTest):
  def test1(self):
    t1 = expr.randn(100, 100)
    t2 = expr.randn(200, 100)
    t3 = expr.add(t1, t2)
    t4 = expr.randn(300, 100)
    t5 = expr.add(t3, t4)
    t6 = expr.randn(400, 100)
    t7 = expr.add(t6, t5)
    t8 = t7.optimized()
    try:
      t8.force()
    except Exception as e:
      print('ignore the exception')

  def test2(self):
    t1 = expr.randn(100, 100)
    t2 = expr.randn(100, 100)
    t3 = expr.dot(t1, t2)
    t4 = expr.randn(200, 100)
    t5 = expr.add(t3, t4)
    t6 = t5.optimized()
    try:
      t6.force()
    except Exception as e:
      print('ignore the exception')

  def test3(self):
    t1 = expr.randn(100, 100)
    t2 = expr.randn(100, 100)
    t3 = expr.add(t1, t2)
    t4 = expr.randn(200, 100)
    t5 = expr.add(t3, t4)
    t6 = expr.randn(100, 100)
    t7 = expr.add(t6, t5)
    t8 = expr.count_zero(t7)
    t9 = t8.optimized()
    try:
      t9.force()
    except Exception as e:
      print('ignore the exception')

