import numpy as np
from spartan import expr, util
from spartan.array import extent

def _svm_disdca_train(X, Y, alpha, w, scale, lambda_n):
  '''
  Using disdca method to train local linear SVM.

  Args:
    X(numpy.ndarray): features of the training data.
    Y(numpy.ndarray): labels of the training data.
    alpha(numpy.array): alpha vector which is the parameter optimized by SVM.
    w(numpy.array): weight vector of the previous iteration.
    scale(int): number of tiles.
    lambda_n: lambda/size(total train data) which is the parameter of this svm model.
  '''
  scaled_lambda = scale * 1.0 / lambda_n
  for i in range(X.shape[0]):
    A = scaled_lambda * np.dot(X[i], X[i])
    B = np.dot(X[i], w)
    dual_ = (Y[i,0] - B)/A
    dual = Y[i,0] * max(0, min(1, Y[i,0] * (dual_ + alpha[i,0]))) - alpha[i,0]
    w = w + (X[i] * scaled_lambda * dual).reshape(w.shape)
    alpha[i,0] = alpha[i,0] + dual
  return alpha

def _svm_mapper(array, ex, labels, alpha, w, lambda_n):
  '''
  Local linear SVM solver.

  Args:
    array(DistArray): features of the training data.
    ex(Extent): Region being processed.
    labels(DistArray): labels of the training data.
    alpha(DistArray): alpha vector which is the parameter optimized by SVM. 
    w(DistArray): weight vector of the previous iteration.
    lambda_n: lambda/size(total train data) which is the parameter of this svm model.
  '''
  X = array.fetch(extent.create((ex.ul[0], 0), (ex.lr[0], array.shape[1]), array.shape))
  Y = labels.fetch(extent.create((ex.ul[0], 0), (ex.lr[0], 1), labels.shape))
  old_alpha = alpha.fetch(extent.create((ex.ul[0], 0), (ex.lr[0], 1), alpha.shape))
  old_w = w[:]
  
  new_alpha = _svm_disdca_train(X, Y, old_alpha, old_w, len(X.tiles), lambda_n)
  
  # update the alpha vector
  yield extent.create((ex.ul[0], 0), (ex.lr[0], 1), alpha.shape), new_alpha
  
def fit(data, labels, T=50, la=1.0):
  '''
  Train an SVM model using the disdca (2013) algorithm.
 
  Args:
    data(Expr): points to be trained.
    labels(Expr): the correct labels of the training data.
    T(int): max training iterations.
    la(float): lambda parameter of this SVM model.
  '''
  w = expr.zeros((data.shape[1], 1), dtype=np.float64)
  alpha = expr.zeros((data.shape[0], 1), dtype=np.float64)
  for i in range(T):
    alpha = expr.shuffle(expr.retile(data, tile_hint=util.calc_tile_hint(data, axis=0)),
                         _svm_mapper, 
                         kw={'labels': labels, 'alpha': alpha, 'w': w, 'lambda_n': la * data.shape[0]},
                         shape_hint=alpha.shape, 
                         cost_hint={ hash(labels) : {'00': 0, '01': np.prod(labels.shape)}, hash(alpha) : {'00': 0, '01': np.prod(alpha.shape)} })
    w = expr.sum(data * alpha * 1.0 / lambda_n, axis=0).reshape((data.shape[1], 1))
    w = w.optimized()
  return w

def predict(w, new_data):
  '''
    Predict the label of the given instance
    
    Args:
      w(DistArray): trained weight vector.
      new_data(Expr or DistArray): data to be predicted
    '''
  ret = np.dot(new_data.glom(), w.glom())
  if ret >= 0: return 1
  else: return -1
  
