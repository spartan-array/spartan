import test_common
import numpy as np
from spartan import expr, util
from spartan.util import Assert

def new_2darray(size_row, size_col):
  #Generate a new random 2d array with dtype = int
  new = np.random.randn(size_row * size_col).reshape(size_row, size_col)
  ret = []

  for row in new:
    tmp = []
    for i in row:
      tmp.append(int(i*100))
    ret.append(tmp)

  return np.array(ret)

class Test_Sort(test_common.ClusterTest):
  """
  def test_sort(self):
    #Sort with axis == 0
    na = new_2darray(10, 10)
    a = expr.from_numpy(na)

    Assert.all_eq(expr.sort(a, 0).glom(),
                  np.sort(na, 0))
  
    #Sort with axis == 1
    nb = new_2darray(10, 10)
    b = expr.from_numpy(nb)

    Assert.all_eq(expr.sort(b, 1).glom(),
                  np.sort(nb, 1))
  
    #Sort with rectangle
    nc = new_2darray(10, 20)
    c = expr.from_numpy(na)

    Assert.all_eq(expr.sort(c, 0).glom(),
                  np.sort(nc, 0))
  
    Assert.all_eq(expr.sort(c, 1).glom(),
                  np.sort(nc, 1))
  """
  def test_argsort(self):
    #Sort with axis == 0
    na = new_2darray(10, 10)
    a = expr.from_numpy(na)

    Assert.all_eq(expr.argsort(a, 0).glom(),
                  np.argsort(na, 0))

    #Sort with axis == 1
    na = new_2darray(10, 10)
    a = expr.from_numpy(na)
    Assert.all_eq(expr.argsort(a, 1).glom(),
                  np.argsort(na, 1))

    #Sort with rectangle
    na = new_2darray(20, 10)
    a = expr.from_numpy(na)

    Assert.all_eq(expr.argsort(a, 0).glom(),
                  np.argsort(na, 0))

    Assert.all_eq(expr.argsort(a, 1).glom(),
                  np.argsort(na, 1))

    util.log_info('\nargsort(a, 0)\nspartan:\n%s\nnumpy:\n%s',  expr.argsort(a, 0).glom(),
								np.argsort(na, 0))
    util.log_info('\nargsort(a, 1)\nspartan:\n%s\nnumpy:\n%s',  expr.argsort(a, 1).glom(),
								np.argsort(na, 1))
