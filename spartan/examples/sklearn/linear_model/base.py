import numpy as np
import spartan
from spartan import expr
from abc import ABCMeta, abstractmethod


def center_data(X, y, fit_intercept, normalize=False):
  """
  Centers data to have mean zero along axis 0. This is here because
  nearly all linear models will want their data to be centered.
  """
  if fit_intercept:
    X_mean = X.mean(axis = 0)
    X -= X_mean
    if normalize:
      X_std = expr.sqrt(expr.sum(X ** 2, axis=0)).force()
      X_std[X_std == 0] = 1
      X /= X_std
    else:
      X_std = expr.ones(X.shape[1])
    
    #problem with broadcast object
    y_mean = y.mean(axis=0)
    #y -= y_mean
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

  _center_data = staticmethod(center_data)

  def predict(self, X):
    return expr.dot(X, self._coef)


class LinearRegression(LinearModel):
  """
  Ordinary least squares Linear Regression.

  Parameters
  ----------
  fit_intercept : boolean, optional
      whether to calculate the intercept for this model. If set
      to false, no intercept will be used in calculations
      (e.g. data is expected to be already centered).

  normalize : boolean, optional, default False
      If True, the regressors X will be normalized before regression.

  Attributes
  ----------
  `coef_` : array, shape (n_features, ) or (n_targets, n_features)
      Estimated coefficients for the linear regression problem.
      If multiple targets are passed during the fit (y 2D), this
      is a 2D array of shape (n_targets, n_features), while if only
      one target is passed, this is a 1D array of length n_features.

  `intercept_` : array
      Independent term in the linear model.

  Notes
  -----
  From the implementation point of view, this is just plain Ordinary
  Least Squares (scipy.linalg.lstsq) wrapped as a predictor object.

  """
  def __init__(self, fit_intercept=True, normalize=False, iterations = 100):
    self.fit_intercept = fit_intercept
    self.normalize = normalize
    self.iterations = iterations

  def fit(self, X, y):
    """ Transform to distarray if it's numpy array"""
    if isinstance(X, np.ndarray):
      X = expr.make_from_numpy(X)
    if isinstance(y, np.ndarray):
      y = expr.make_from_numpy(y)
    
    X, y, X_mean, y_mean, X_std = self._center_data(
        X, y, self.fit_intercept, self.normalize)
    
    N_DIM = X.shape[1]
    self._coef = np.random.randn(N_DIM, 1) 

    for i in range(self.iterations):
      yp = expr.dot(X, self._coef)
      diff = X * (yp - y)
      grad = expr.sum(diff, axis=0).glom().reshape((N_DIM, 1))
      self._coef = self._coef - grad * 1e-6  
    return self
