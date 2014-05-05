import numpy as np
import spartan
from spartan import expr
from spartan.examples import sgd
from spartan.array import distarray

class RidgeRegression(sgd.SGDRegressor):
  def __init__(self, x, y, ridge_lambda, iterations, alpha = 1e-6):
    super(RidgeRegression, self).__init__(x, y, iterations, alpha)
    self.ridge_lambda = ridge_lambda

  def update(self):
    '''
    gradient_update = 2xTxw - 2xTy + 2* lambda * w
    Correct this if the update function is wrong.
    '''
    xT = expr.transpose(self.x)
    g1 = expr.dot(expr.dot(xT, self.x), self.w)
    g2 = expr.dot(xT, self.y)
    g3 = self.ridge_lambda * self.w
    g4 = (g1 + g2 + g3)
    return expr.reshape(g4, (1, self.N_DIM))

def ridge_regression(x, y, ridge_lambda, iterations):
  ridge_reg = RidgeRegression(x, y, ridge_lambda, iterations)
  return ridge_reg.train()

def run(N_EXAMPLES, N_DIM, iterations):
  x = expr.eager(expr.rand(N_EXAMPLES, N_DIM, tile_hint=(N_EXAMPLES / 10, 10)))
  y = expr.eager(expr.rand(N_EXAMPLES, 1, tile_hint=(N_EXAMPLES / 10, 1)))
  ridge_regression(x, y, 1, iterations)

