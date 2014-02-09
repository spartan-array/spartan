import numpy as np
import spartan
from spartan import expr
from abc import ABCMeta, abstractmethod


def center_data(X, y, fit_intercept, normalize=False, copy=True):
  """
  Centers data to have mean zero along axis 0. This is here because
  nearly all linear models will want their data to be centered.
  """
  if fit_intercept:
    X_mean = X.mean(axis = 0)
    X -= X_mean
    if normalize:
      X_std = expr.sqrt(expr.sum(X ** 2, axis=0))
      X_std[X_std == 0] = 1
      X /= X_std
    else:
      X_std = expr.ones(X.shape[1])

    y_mean = y.mean(axis=0)
    y = y - y_mean
  else:
    X_mean = expr.zeros(X.shape[1])
    X_std = expr.ones(X.shape[1])
    y_mean = 0. if y.ndim == 1 else expr.zeros(y.shape[1], dtype=X.dtype)
  return X, y, X_mean, y_mean, X_std


class LinearModel(object):
  """ Base class for Linear Models """
  __metaclass__ = ABCMeta
  @abstractmethod
  def fit(self, X, y):
    raise NotImplementedError


class LinearRegression(LinearModel):
  def __init__(self, fit_intercept=True, normalize=False, copy_X=True):
    self.fit_intercept = fit_intercept
    self.normalize = normalize
    self.copy_X = copy_X

  def fit(self, X, y):
    if isinstance(X, np.ndarray):
      X = expr.make_from_numpy(X)
    if isinstance(y, np.ndarray):
      y = expr.make_from_numpy(y)
    
    N_DIM = X.shape[1]
    self._coef = np.random.randn(N_DIM, 1)

    for i in range(100):
      yp = expr.dot(X, self._coef)
      diff = X * (yp - y)
      print expr.sum(diff).force().glom()
      grad = expr.sum(diff, axis=0).glom().reshape((N_DIM, 1))
      self._coef = self._coef - grad * 1e-6 
    return self


