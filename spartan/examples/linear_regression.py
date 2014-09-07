import numpy as np
import spartan
from spartan import expr
from spartan.examples import sgd

class LinearRegression(sgd.SGDRegressor):
  def __init__(self, x, y, iterations, alpha = 1e-6):
    super(LinearRegression, self).__init__(x, y, iterations, alpha)

  def update(self):
    '''
    gradient_update = (h(w) - y) * x
    h(w) = x * w
    '''
    yp = expr.dot(self.x, self.w)
    return self.x * (yp - self.y)

def linear_regression(x, y, iterations):
  lreg = LinearRegression(x, y, iterations)
  return lreg.train()

def run(N_EXAMPLES, N_DIM, iterations):
  x = expr.rand(N_EXAMPLES, N_DIM)
  y = expr.rand(N_EXAMPLES, 1)
  linear_regression(x, y, iterations)
