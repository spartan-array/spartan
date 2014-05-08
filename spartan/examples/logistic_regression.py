import numpy as np
import spartan
from spartan import expr
from spartan.examples import sgd

class LogisticRegression(sgd.SGDRegressor):
  def __init__(self, x, y, iterations, alpha = 1e-6):
    super(LogisticRegression, self).__init__(x, y, iterations, alpha)

  def update(self):
    '''
    gradient_update = (h(w) - y) * x
    h(w) = 1 / (1 + e^(-(x*w)))
    '''
    g = expr.exp(expr.dot(self.x, self.w))
    yp = g / (g + 1)
    return self.x * (yp - self.y)

def logistic_regression(x, y, iterations):
  logreg = LogisticRegression(x, y, iterations)
  return logreg.train()

def run(N_EXAMPLES, N_DIM, iterations):
  x = expr.eager(expr.rand(N_EXAMPLES, N_DIM, tile_hint=(N_EXAMPLES / 10, 10)))
  y = expr.eager(expr.rand(N_EXAMPLES, 1, tile_hint=(N_EXAMPLES / 10, 1)))
  logistic_regression(x, y, iterations)

