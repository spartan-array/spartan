import test_common
import numpy as np
from spartan import expr
from spartan.util import Assert

class Test_NewAxis(test_common.ClusterTest):
  def test_newaxis(self):
    na = np.arange(100).reshape(10, 10)
    a = expr.from_numpy(na)

    print a.shape

    Assert.all_eq( na[np.newaxis, 2:7, 4:8].shape,
                   a[expr.newaxis,2:7, 4:8].shape)

    Assert.all_eq( na[np.newaxis, 2:7, np.newaxis, 4:8].shape,
                   a[expr.newaxis,2:7, expr.newaxis, 4:8].shape)

    Assert.all_eq( na[np.newaxis, 2:7, np.newaxis, 4:8, np.newaxis].shape,
                   a[expr.newaxis,2:7, expr.newaxis, 4:8, expr.newaxis].shape)

    #Extreme case
    Assert.all_eq( na[np.newaxis, np.newaxis, np.newaxis, np.newaxis, 2:7, np.newaxis, np.newaxis, np.newaxis, 4:8, np.newaxis, np.newaxis, np.newaxis].shape,
                   a[expr.newaxis, expr.newaxis, expr.newaxis, expr.newaxis, 2:7, expr.newaxis, expr.newaxis, expr.newaxis, 4:8, expr.newaxis, expr.newaxis, expr.newaxis].shape)
