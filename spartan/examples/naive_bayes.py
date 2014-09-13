import numpy as np
from spartan import expr, util
from spartan.array import extent, distarray
  
def _sum_instance_by_label_mapper(array, ex, labels, label_size):
  '''
  For each label, compute the sum of the feature vectors which belong to that label.
  
  Args:
    array(DistArray): tf-idf normalized training data.
    ex(Extent): Region being processed.
    labels(DistArray): labels of the training data.
    label_size: the number of different labels.
  '''
  X = array.fetch(extent.create((ex.ul[0], 0), (ex.lr[0], array.shape[1]), array.shape))
  Y = labels.fetch(extent.create((ex.ul[0], 0), (ex.lr[0], 1), labels.shape))
  
  sum_instance_by_label = np.zeros((label_size, X.shape[1]))
  for i in xrange(Y.shape[0]):
    sum_instance_by_label[Y[i, 0]] += X[i]
    
  yield extent.create((0, 0), (label_size, X.shape[1]), (label_size, X.shape[1])), sum_instance_by_label
  
def fit(data, labels, label_size, alpha=1.0):
  '''
  Train standard naive bayes model.
 
  Args:
    data(Expr): documents to be trained.
    labels(Expr): the correct labels of the training data.
    label_size(int): the number of different labels.
    alpha(float): alpha parameter of naive bayes model.
  '''
  # calc document freq
  df = expr.reduce(data,
                   axis=0,
                   dtype_fn=lambda input: input.dtype,
                   local_reduce_fn=lambda ex, data, axis: (data > 0).sum(axis),
                   accumulate_fn=np.add)
  
  idf = expr.log(data.shape[0] * 1.0 / (df + 1)) + 1
   
  # Normalized Frequency for a feature in a document is calculated by dividing the feature frequency 
  # by the root mean square of features frequencies in that document
  square_sum = expr.reduce(data,
                           axis=1,
                           dtype_fn=lambda input: input.dtype,
                           local_reduce_fn=lambda ex, data, axis: np.square(data).sum(axis),
                           accumulate_fn=np.add)
  
  rms = expr.sqrt(square_sum * 1.0 / data.shape[1])
  
  # calculate weight normalized Tf-Idf
  data = data / rms.reshape((data.shape[0], 1)) * idf.reshape((1, data.shape[1]))
  
  # add up all the feature vectors with the same labels
  #weights_per_label_and_feature = expr.ndarray((label_size, data.shape[1]), dtype=np.float64)
  #for i in range(label_size):
  #  i_mask = (labels == i)
  #  weights_per_label_and_feature = expr.assign(weights_per_label_and_feature, np.s_[i, :], expr.sum(data[i_mask, :], axis=0))
  weights_per_label_and_feature = expr.shuffle(expr.retile(data, tile_hint=util.calc_tile_hint(data, axis=0)),
                                               _sum_instance_by_label_mapper,
                                               target=expr.ndarray((label_size, data.shape[1]), dtype=np.float64, reduce_fn=np.add),
                                               kw={'labels': labels, 'label_size': label_size},
                                               cost_hint={hash(labels):{'00':0, '01':np.prod(labels.shape)}})

  # sum up all the weights for each label from the previous step
  weights_per_label = expr.sum(weights_per_label_and_feature, axis=1)
  
  # generate naive bayes per_label_and_feature weights
  weights_per_label_and_feature = expr.log((weights_per_label_and_feature + alpha) / 
                                           (weights_per_label.reshape((weights_per_label.shape[0], 1)) + 
                                            alpha * weights_per_label_and_feature.shape[1]))

  return {'scores_per_label_and_feature': weights_per_label_and_feature.optimized().force(),
          'scores_per_label': weights_per_label.optimized().force(),
          }
  
def predict(model, new_data):
  '''
  Predict the label of the given instance.
  
  Args:
    model(dict): trained naive bayes model.
    new_data(Expr or DistArray): data to be predicted
  '''
  scores_per_label_and_feature = model['scores_per_label_and_feature']

  scoring_vector = expr.dot(scores_per_label_and_feature, expr.transpose(new_data))
  # util.log_warn('scoring_vector:%s', scoring_vector.glom().T)  
  
  return np.argmax(scoring_vector.glom())
  
