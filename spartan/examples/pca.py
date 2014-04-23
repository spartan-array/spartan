import spartan
from spartan import expr, array
import numpy as np
from .ssvd import svd

class PCA(object):
  """Principal component analysis (PCA)

  Currently we implement this based on lanczos SVD. lanczos SVD 
  is efficient for decomposing sparse matrix.
  """
  def __init__(self, n_components=None):
    self.n_components = n_components

  def fit(self, X, rank=None):
    """Fit the model to the data X.

    Parameters
    ----------
    X:  Spartan distributed array of shape (n_samples, n_features).

    rank: Integer, optinal(default=None), the rank of this matrix. 

    Returns
    -------
    self : object
        Returns the instance itself.
    """    
    self.mean_ = expr.mean(X, axis=0)
    X -= self.mean_
    X = X.force()
    if rank is None:
      rank = min(X.shape[0], X.shape[1])

    V, S, U = svd(X, rank)
    self.components_ = U
    self.components_ = self.components_[:self.n_components, :]
    return self

  def transform(self, X):
    """Reduce dimensions of matrix X.

    Parameters
    ----------
    X : Spartan distributed array of shape (n_samples, n_features).

    Returns
    -------
    X_new : reduced dimension numpy.array, shape (n_samples, n_components)

    """
    X_transformed = X - self.mean_
    X_transformed = expr.dot(X_transformed, self.components_.T).glom()
    return X_transformed

  def inverse_transform(self, X):
    """Transform data back to its original space.

    Parameters
    ----------
    X : spartan arary or numpy array of shape (n_samples, n_components).

    Returns
    X_original: numpyarray of shape (n_samples, n_features). 
    """
    if isinstance(X, expr.Expr) or isinstance(X, array.distarray.DistArray):
      return (expr.dot(X, self.components_) + self.mean_).force()
    else:
      return np.dot(X, self.components_) + self.mean_.glom()
