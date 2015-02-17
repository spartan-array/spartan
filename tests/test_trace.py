from spartan import expr
from spartan.util import Assert
from spartan import util
from spartan.array import extent
from spartan.config import FLAGS

import numpy as np
import test_common
import sys
import os

from scipy import sparse as sp


class RedirectStdStreams(object):
  def __init__(self, stdout=None, stderr=None):
    # Comment these two lines if you want to see trace result of this test.
    self._stdout = stdout or sys.stdout
    self._stderr = stderr or sys.stderr
    pass

  def __enter__(self):
    self.old_stdout, self.old_stderr = sys.stdout, sys.stderr
    self.old_stdout.flush()
    self.old_stderr.flush()
    sys.stdout, sys.stderr = self._stdout, self._stderr

  def __exit__(self, exc_type, exc_value, traceback):
    self._stdout.flush()
    self._stderr.flush()
    sys.stdout = self.old_stdout
    sys.stderr = self.old_stderr


# Since nobody will check the output of OK unittest,
# capture it to make output cleaerer.
class TestTrace(test_common.ClusterTest):
  def setUp(self):
    FLAGS.capture_expr_stack = 1
    FLAGS.opt_keep_stack = 1

  def tearDown(self):
    FLAGS.capture_expr_stack = 0
    FLAGS.opt_keep_stack = 0

  def test_trace(self):
    devnull = open(os.devnull, 'w')
    with RedirectStdStreams(stdout=devnull, stderr=devnull):
      t1 = expr.randn(100, 100)
      t2 = expr.randn(200, 100)
      t3 = expr.add(t1, t2)
      t4 = expr.randn(300, 100)
      t5 = expr.add(t3, t4)
      t6 = expr.randn(400, 100)
      t7 = expr.add(t6, t5)
      t8 = t7.optimized()
      try:
        t8.evaluate()
      except Exception as e:
        pass

      t1 = expr.randn(100, 100)
      t2 = expr.randn(100, 100)
      t3 = expr.dot(t1, t2)
      t4 = expr.randn(200, 100)
      t5 = expr.add(t3, t4)
      t6 = t5.optimized()
      try:
        t6.evaluate()
      except Exception as e:
        pass

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
        t9.evaluate()
      except Exception as e:
        pass


if __name__ == '__main__':
  import unittest
  unittest.main()
