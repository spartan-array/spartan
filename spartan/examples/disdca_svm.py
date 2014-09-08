import numpy as np
from spartan import expr, util, blob_ctx
from spartan.array import extent

def _svm_disdca_train(X, Y, alpha, w, m, scale, lambda_n):
  '''
  Using disdca method to train local linear SVM.

  Args:
    X(numpy.ndarray): features of the training data.
    Y(numpy.ndarray): labels of the training data.
    alpha(numpy.array): alpha vector which is the parameter optimized by SVM.
    w(numpy.array): weight vector of the previous iteration.
    m(int): number of samples to train (now we set it to the whole local data set).
    scale(int): number of tiles.
    lambda_n: lambda/size(total train data) which is the parameter of this svm model.
  '''
  scaled_lambda = scale * 1.0 / lambda_n
  for i in range(m):
    A = scaled_lambda * np.dot(X[i], X[i])
    B = np.dot(X[i], w)
    dual_ = (Y[i,0] - B)/A
    dual = Y[i,0] * max(0, min(1, Y[i,0] * (dual_ + alpha[i,0]))) - alpha[i,0]
    w = w + (X[i] * scaled_lambda * dual).reshape(w.shape)
    alpha[i,0] = alpha[i,0] + dual
  new_w = (X * alpha * scaled_lambda).sum(axis=0).reshape(w.shape)
  return new_w, alpha

def _svm_mapper(array, ex, labels, alpha, w, m, scale, lambda_n):
  '''
  Local linear SVM solver.

  Args:
    array(DistArray): features of the training data.
    ex(Extent): Region being processed.
    labels(DistArray): labels of the training data.
    alpha(DistArray): alpha vector which is the parameter optimized by SVM. 
    w(DistArray): weight vector of the previous iteration.
    m(int): number of samples to train (now we set it to the whole local data set).
    scale(int): number of tiles
    lambda_n: lambda/size(total train data) which is the parameter of this svm model.
  '''
  X = array.fetch(extent.create((ex.ul[0], 0), (ex.lr[0], array.shape[1]), array.shape))
  Y = labels.fetch(extent.create((ex.ul[0], 0), (ex.lr[0], 1), labels.shape))
  
  tile_id = ex.ul[0]/(ex.lr[0]-ex.ul[0])
  ex_alpha = extent.create((tile_id*m, 0), ((tile_id+1)*m, 1), alpha.shape)
  old_alpha = alpha.fetch(ex_alpha)
  
  old_w = np.zeros((X.shape[1],1)) if w is None else w[:]
  
  new_w, new_alpha = _svm_disdca_train(X, Y, old_alpha, old_w, m, scale, lambda_n)
  
  # update the alpha vector
  alpha.update(ex_alpha, new_alpha)
  
  # reduce the weight vector
  yield extent.create((0,0),(array.shape[1],1),(array.shape[1], 1)), new_w
  
def fit(data, labels, T=50, la=1.0):
  '''
  Train an SVM model using the disdca (2013) algorithm.
 
  Args:
    data(Expr): points to be trained.
    labels(Expr): the correct labels of the training data.
    T(int): max training iterations.
    la(float): lambda parameter of this SVM model.
  '''
  w = None
  num_workers = blob_ctx.get().num_workers
  m = data.shape[0] / num_workers
  alpha = expr.zeros((data.shape[0], 1), dtype=np.float64).optimized().force()
  for i in range(T):
    new_weight = expr.shuffle(data, _svm_mapper, target=expr.ndarray((data.shape[1], 1), dtype=np.float64, reduce_fn=np.add), 
                              kw={'labels': labels, 'alpha': alpha, 'w': w, 'm': m, 'scale': num_workers, 'lambda_n': la * data.shape[0]}, 
                              cost_hint={ hash(labels) : {'00': 0, '01': np.prod(labels.shape)} })
    w = new_weight / num_workers
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
  
