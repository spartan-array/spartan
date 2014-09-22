import test_common
import numpy as np
from spartan import expr, util
from spartan.util import Assert

class Test_NewAxis(test_common.ClusterTest):
  def test_newaxis(self):
    na = np.arange(100).reshape(10, 10)
    a = expr.from_numpy(na)

    Assert.all_eq(na[np.newaxis, 2:7, 4:8].shape,
                  a[expr.newaxis,2:7, 4:8].shape)

    Assert.all_eq(na[np.newaxis, 2:7, np.newaxis, 4:8].shape,
                  a[expr.newaxis,2:7, expr.newaxis, 4:8].shape)

    Assert.all_eq(na[np.newaxis, 2:7, np.newaxis, 4:8, np.newaxis].shape,
                  a[expr.newaxis,2:7, expr.newaxis, 4:8, expr.newaxis].shape)

    #Extreme case
    Assert.all_eq(na[np.newaxis, np.newaxis, np.newaxis, np.newaxis, 2:7, 
		  np.newaxis, np.newaxis, np.newaxis, 4:8, np.newaxis, 
		  np.newaxis, np.newaxis].shape,

                  a[expr.newaxis, expr.newaxis, expr.newaxis, expr.newaxis,
                  2:7, expr.newaxis, expr.newaxis, expr.newaxis, 4:8,
                  expr.newaxis, expr.newaxis, expr.newaxis].shape)

    util.log_info('\na.shape:  %s  \nna.shape: %s',
		  a[expr.newaxis,2:7, expr.newaxis, 4:8, expr.newaxis,
			expr.newaxis, expr.newaxis].shape,

                  na[np.newaxis, 2:7, np.newaxis, 4:8, np.newaxis,
			np.newaxis, np.newaxis].shape)

  def test_del_dim(self):
    na = np.arange(100).reshape(10, 10)
    a = expr.from_numpy(na)

    Assert.all_eq(na[2:7, 8], a[2:7, 8].glom())
    Assert.all_eq(na[3:9, 4].shape, a[3:9, 4].shape)

    Assert.all_eq(na[2:7, -1], a[2:7, -1].glom())
    Assert.all_eq(na[-1, 3:9].shape, a[-1, 3:9].shape)

    util.log_info('\na.shape: %s \nna.shape %s', a[3:9, 4].shape, na[3:9, 4].shape)

  def test_combo(self):
    na = np.arange(100).reshape(10, 10)
    a = expr.from_numpy(na)

    Assert.all_eq(na[np.newaxis, 2:7, 4],
                  a[expr.newaxis, 2:7, 4].glom())
    Assert.all_eq(na[2:7, np.newaxis, -1],
                  a[2:7, expr.newaxis, -1].glom())
    Assert.all_eq(na[-1, np.newaxis, 2:7],
                  a[-1, expr.newaxis, 2:7].glom())
    Assert.all_eq(na[np.newaxis, 2:7, np.newaxis, np.newaxis, 4, np.newaxis, np.newaxis],
                  a[expr.newaxis, 2:7, expr.newaxis, expr.newaxis, 4, expr.newaxis, expr.newaxis].glom())

    util.log_info('\na.shape:  %s \nna.shape: %s',
		a[expr.newaxis, 2:7, expr.newaxis, expr.newaxis, -1, expr.newaxis, expr.newaxis].shape,
                na[np.newaxis, 2:7, np.newaxis, np.newaxis, -1, np.newaxis, np.newaxis].shape)
