import numpy as np
import spartan
from spartan import expr, core, blob_ctx, array
from sklearn.ensemble import RandomForestClassifier as SKRF

def _build_mapper(ex, 
                  X, 
                  y, 
                  criterion,
                  max_depth,
                  min_samples_split,
                  min_samples_leaf,
                  max_features,
                  bootstrap):
  """ mapper kernel for building classifier forest """
  n_estimators = ex.shape[0]
  X = X.glom()
  y = y.glom()
  rf = SKRF(n_estimators = n_estimators,
                              criterion = criterion,
                              max_depth = max_depth,
                              n_jobs = 1,
                              min_samples_split = min_samples_split,
                              min_samples_leaf = min_samples_leaf,
                              max_features = max_features,
                              bootstrap = bootstrap)

  rf.fit(X, y) 
  result = core.LocalKernelResult()
  result.result = rf
  return result

class RandomForestClassifier(object):
  """A random forest classifier.

  A random forest is a meta estimator that fits a number of decision tree
  classifiers on various sub-samples of the dataset and use averaging to
  improve the predictive accuracy and control over-fitting.

  Parameters
  ----------
  n_estimators : integer, optional (default=10)
      The number of trees in the forest.

  criterion : string, optional (default="gini")
      The function to measure the quality of a split. Supported criteria are
      "gini" for the Gini impurity and "entropy" for the information gain.
      Note: this parameter is tree-specific.

  max_features : int, float, string or None, optional (default="auto")
      The number of features to consider when looking for the best split:
        - If int, then consider `max_features` features at each split.
        - If float, then `max_features` is a percentage and
          `int(max_features * n_features)` features are considered at each
          split.
        - If "auto", then `max_features=sqrt(n_features)`.
        - If "sqrt", then `max_features=sqrt(n_features)`.
        - If "log2", then `max_features=log2(n_features)`.
        - If None, then `max_features=n_features`.

      Note: this parameter is tree-specific.

  max_depth : integer or None, optional (default=None)
      The maximum depth of the tree. If None, then nodes are expanded until
      all leaves are pure or until all leaves contain less than
      min_samples_split samples.
      Ignored if ``max_samples_leaf`` is not None.
      Note: this parameter is tree-specific.

  min_samples_split : integer, optional (default=2)
      The minimum number of samples required to split an internal node.
      Note: this parameter is tree-specific.

  min_samples_leaf : integer, optional (default=1)
      The minimum number of samples in newly created leaves.  A split is
      discarded if after the split, one of the leaves would contain less then
      ``min_samples_leaf`` samples.
      Note: this parameter is tree-specific.

  max_leaf_nodes : int or None, optional (default=None)
      Grow trees with ``max_leaf_nodes`` in best-first fashion.
      Best nodes are defined as relative reduction in impurity.
      If None then unlimited number of leaf nodes.
      If not None then ``max_depth`` will be ignored.
      Note: this parameter is tree-specific.

  bootstrap : boolean, optional (default=True)
      Whether bootstrap samples are used when building trees.
  """
  def __init__(self,
               n_estimators=10,
               criterion="gini",
               max_depth=None,
               min_samples_split=2,
               min_samples_leaf=1,
               max_features="auto",
               max_leaf_nodes=None,
               bootstrap=True):
    self.n_estimators = n_estimators
    self.criterion = criterion
    self.max_depth = max_depth
    self.min_samples_split = min_samples_split
    self.min_samples_leaf = min_samples_leaf
    self.max_features = max_features
    self.max_leaf_nodes = max_leaf_nodes
    self.bootstrap = bootstrap
    self.forests = None

  def fit(self, X, y):
    """
    Parameters
    ----------
    X : array-like of shape = [n_samples, n_features]
        The training input samples.

    y : array-like, shape = [n_samples] or [n_samples, n_outputs]
        The target values (integers that correspond to classes in
        classification, real numbers in regression).

    Returns
    -------
    self : object
        Returns self.
    """
    if isinstance(X, np.ndarray):
      X = expr.from_numpy(X)
    if isinstance(y, np.ndarray):
      y = expr.from_numpy(y)
    
    self.n_classes = np.unique(y.glom()).size
    ctx = blob_ctx.get()
    n_workers = ctx.num_workers
    
    if self.n_estimators < n_workers:
      tile_hint = (1, 1)
    else:
      tile_hint = (self.n_estimators / n_workers, 1)
    
    """
    task_array is used for deciding how many trees each worker needs to train.
    e.g. If we train 20 trees on 5 workers. Then the tile looks like: 
          (0:4, 0:1), (4:8, 0:1), (8:12, 0:1), (12:16, 0:1), (16:20, 0:1)
    """
    task_array = expr.ndarray((self.n_estimators, 1), tile_hint=tile_hint)
    task_array = task_array.force()

    X = X.force()
    y = y.force()
    results = task_array.foreach_tile(mapper_fn = _build_mapper, 
                                      kw = {'X' : X, 'y' : y, 
                                            'criterion' : self.criterion,
                                            'max_depth' : self.max_depth,
                                            'min_samples_split' : self.min_samples_split,
                                            'min_samples_leaf' : self.min_samples_leaf,
                                            'max_features' : self.max_features,
                                            'bootstrap' : self.bootstrap})
    """ Aggregate the result """
    self.forests = [v for k, v in results.iteritems()]

  def predict(self, X):
    if isinstance(X, expr.Expr) or isinstance(X, array.distarray.DistArray):
      X = X.glom()
  
    sk_proba = np.zeros((X.shape[0], self.n_classes), np.float64)
    
    """ let each forest predict, then add the probabilities together. """
    if self.forests is not None:
      for f in self.forests:
        sk_proba += f.predict_proba(X) * len(f.estimators_)
    
    """ choose the most probably one """ 
    result = np.array([np.argmax(sk_proba[i]) for i in xrange(sk_proba.shape[0])])
    return result

  def score(self, X, y):
    """Return the mean accuracy on the given test data and labels.

    Parameters
    ----------
    X : array-like, shape = (n_samples, n_features)
        Test samples.

    y : array-like, shape = (n_samples,)
        True labels for X.

    Returns
    -------
    score : float
        Mean accuracy of self.predict(X) wrt. y.
    """     
    if not isinstance(y, np.ndarray):
      y = y.glom()
    return np.mean(self.predict(X) == y)
