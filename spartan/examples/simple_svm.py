import numpy as np
from random import randint
from spartan import expr, util
import spartan
import math
from spartan import blob_ctx

def margin_mapper(ex, alpha_data, axis, label, data):
  '''
  Compute the linear SVM margin for the given extent.

  Args:
    ex(Extent): Region being processed.
    alpha_data(numpy.array): alpha vector which is the parameter optimized by SVM 
    axis(int): Unused.
    label(DistArray): the labels of the training data
    data(DistArray): the data need to be doted 
  '''
  
  label_data = label.fetch(ex)
  data_data = data.fetch(ex)
  tmp = np.multiply(alpha_data, label_data)
  result = np.dot(tmp[:,0], data_data[:,0])
  return result
  
class SVM:
  '''
  A simple implementation for Support Vector Machine
  
  Args:
    C(float64): optional. C parameter in C-Support Vector Classification. 1 by default. 
    tol(float64): optional. Numeric stopping criterion (WRITEME).
    maxiter(int): optional. Stop solver after this many iterations regardless of accuracy
  '''
  
  def __init__(self, C=1.0, tol=1e-6, maxiter=50):
    self.C = C
    self.tol = tol
    self.maxiter = maxiter
    self.usew_ = False
    self.b = 0.0
    self.ctx = blob_ctx.get()
    
  # Calculate margin of given instance
  def margin_one(self, arr):
    '''
    Calculate margin of given instance
    
    Args:
      arr(Expr): data to be calculated
    '''
    
    f = self.b

    if self.usew_:
      f += expr.dot(arr, self.w).glom()
      
    return f      
      
  def predict_one(self, arr):
    '''
    Predict the label of the given instance
    
    Args:
      arr(Expr): data to be predicted
    '''
    
    if self.margin_one(arr) > self.tol:
      return 1
    else:
      return -1

  # Calculate margin of given instances
  def margins(self, data):
    '''
    Calculate margin of given instances
    
    Args:
      data(Expr): data to be calculated
    '''
    
    return expr.dot(data, self.w) + self.b

  def predict(self, data):
    '''
    Predict the label of the given instances
    
    Args:
      data(Expr): data to be predicted
    '''
    
    margs = self.margins(data).force()
    for i in xrange(margs.shape[0]):
      if margs[i,0] > self.tol:
        margs[i,0] = 1
      else:
        margs[i,0] = -1

    return margs
  
  def take_step(self, i, j, N, labels, kernel_results):
    '''
    Perform one optimization step. Updates model with respect to to training example i, j
    Modifies self.E to reflect the error of new model to labels 
    
    Args:
      i(int): index of the alpha to be optimized
      j(int): index of the alpha to be optimized
      N(int): the number of features
      labels(Expr): the labels of the training data
      kernel_results(Expr): the result of the kernel function on the training data
    '''
    
    if i == j: return 0
    
    ai = self.alpha[i,0]
    aj = self.alpha[j,0]
    Ei = self.E[i,0]
    Ej = self.E[j,0]
    yi = labels[i,0]
    yj = labels[j,0]
    kii = kernel_results[i,i].glom()
    kjj = kernel_results[j,j].glom()
    kij = kernel_results[i,j].glom()
    
    s = yi * yj
    
    L = 0
    H = self.C
    if yi == yj:
      L = max(0, ai+aj-self.C)
      H = min(self.C, ai+aj)
    else:
      L = max(0, aj-ai)
      H = min(self.C, self.C+aj-ai)

    if abs(L-H) < self.tol:
      return 0

    eta = kii + kjj - 2 * kij
    if eta > self.tol:
      newaj = aj + yj * (Ei-Ej) / eta
      if newaj > H: newaj = H
      if newaj < L: newaj = L
    else:
      return 0
    
    if abs(aj - newaj) < self.tol*(aj+newaj+self.tol):
      return 0
      
    newai = ai + s * (aj - newaj)
    
    b1 = self.b - Ei - yi * (newai-ai) * kii - yj * (newaj-aj) * kij
    b2 = self.b - Ej - yi * (newai-ai) * kij - yj * (newaj-aj) * kjj
    self.b = 0.5*(b1+b2)
    if (newai > self.tol) and (newai < self.C):
      self.b = b1
    if (newaj > self.tol) and (newaj < self.C):
      self.b = b2
    
    self.b = self.b[0,0]
    self.alpha[i,0] = newai
    self.alpha[j,0] = newaj 
    
    for i in xrange(N):
      self.E[i,0] = self.b + expr.reduce(self.alpha, axis=None, dtype_fn=lambda input: input.dtype,
                                         local_reduce_fn=margin_mapper,
                                         accumulate_fn=np.add, 
                                         fn_kw=dict(label=labels, data=kernel_results[:,i].force())).glom() - labels[i,0]
    #print 'b', self.b
    #print 'alpha', self.alpha.glom().T
    #print 'E', self.E.glom().T
    print 'success opt i, j:', i, j
    return 1
 
  def examine_example(self, i, N, labels, kernel_results):
    ''' Check if the alpha_i can be optimized. It should satisfy the KKT condition.
    If so, choose it as one parameter to be optimized.
    
    Args:
      i(int): index of the alpha to be checked
      N(int): the number of features
      labels(Expr): the labels of the training data
      kernel_results(Expr): the result of the kernel function on the training data
    '''
    
    Ei = self.E[i,0]
    ai = self.alpha[i,0]
    r = labels[i,0] * Ei
    # check if satisfy KKT condition
    if r < -self.tol and ai < self.C or r > self.tol and ai > self.tol:
      alpha_expr = expr.lazify(self.alpha)
      active_set_mask = (alpha_expr > self.tol) * (alpha_expr < self.C)
      active_set = active_set_mask.glom().nonzero()[0]
      #print 'actives:', active_set
      
      # first check the jth example that maximize the |Ei - Ej|
      idxj = -1
      if active_set.shape[0] > 1:
        active_E = expr.abs(expr.lazify(self.E) - Ei)[active_set_mask-True]
        idxj = np.argmax(active_E.glom())

        if self.take_step(idxj, i, N, labels, kernel_results): return 1
      
      # then check the examples in active_set
      for j in active_set:
        if j != idxj and self.take_step(j, i, N, labels, kernel_results): return 1
      
      # finally check the other examples  
      for j in xrange(N):
        if j not in active_set and self.take_step(j, i, N, labels, kernel_results): return 1
      
    return 0
          
  def train_smo_1998(self, data, labels):
    '''
    Train an SVM model using the SMO (1998) algorithm.
   
    Args:
      data(Expr): points to be trained
      labels(Expr): the correct labels of the training data
    '''
    
    N = data.shape[0] # Number of instances
    D = data.shape[1]  # Number of features

    self.b = 0.0
    self.alpha = expr.zeros((N,1), dtype=np.float64, tile_hint=[N/self.ctx.num_workers, 1]).force()
    
    # linear kernel
    kernel_results = expr.dot(data, expr.transpose(data), tile_hint=[N/self.ctx.num_workers, N])   
    
    labels = expr.force(labels)
    self.E = expr.zeros((N,1), dtype=np.float64, tile_hint=[N/self.ctx.num_workers, 1]).force()
    for i in xrange(N):
      self.E[i, 0] = self.b + expr.reduce(self.alpha, axis=None, dtype_fn=lambda input: input.dtype,
                                          local_reduce_fn=margin_mapper,
                                          accumulate_fn=np.add, 
                                          fn_kw=dict(label=labels, data=kernel_results[:,i].force())).glom() - labels[i, 0]
    
    util.log_info("Starting SMO")
    it = 0
    num_changed = 0
    examine_all = True
    while (num_changed > 0 or examine_all) and (it < self.maxiter):
      util.log_info("Iteration:%d", it)

      num_changed = 0
      
      if examine_all:
        for i in xrange(N): 
          num_changed += self.examine_example(i, N, labels, kernel_results)
      else:
        for i in xrange(N):
          if self.alpha[i, 0] > 0 and self.alpha[i, 0] < self.C:
            num_changed += self.examine_example(i, N, labels, kernel_results)

      it += 1

      if examine_all: examine_all = False
      elif num_changed == 0: examine_all = True
    
    self.w = expr.zeros((D, 1), dtype=np.float64).force()
    for i in xrange(D): 
      self.w[i,0] = expr.reduce(self.alpha, axis=None, dtype_fn=lambda input: input.dtype,
                              local_reduce_fn=margin_mapper,
                              accumulate_fn=np.add, 
                              fn_kw=dict(label=labels, data=expr.force(data[:,i]))).glom()
    self.usew_ = True
    print 'iteration finish:', it
    print 'b:', self.b
    print 'w:', self.w.glom()
  
  
  def train_smo_2005(self, data, labels):
    '''
    Train an SVM model using the SMO (2005) algorithm.
   
    Args:
      data(Expr): points to be trained
      labels(Expr): the correct labels of the training data
    '''
    
    N = data.shape[0] # Number of instances
    D = data.shape[1]  # Number of features

    self.b = 0.0
    alpha = expr.zeros((N,1), dtype=np.float64, tile_hint=[N/self.ctx.num_workers, 1]).force()
    
    # linear kernel
    kernel_results = expr.dot(data, expr.transpose(data), tile_hint=[N/self.ctx.num_workers, N])
    gradient = expr.ones((N, 1), dtype=np.float64, tile_hint=[N/self.ctx.num_workers, 1]) * -1.0
    
    expr_labels = expr.lazify(labels)
    
    util.log_info("Starting SMO")
    pv1 = pv2 = -1
    it = 0
    while it < self.maxiter:
      util.log_info("Iteration:%d", it)
      
      minObj = 1e100
      
      expr_alpha = expr.lazify(alpha)
      G = expr.multiply(labels, gradient) * -1.0

      v1_mask = ((expr_labels > self.tol) * (expr_alpha < self.C) + (expr_labels < -self.tol) * (expr_alpha > self.tol))
      v1 = expr.argmax(G[v1_mask-True]).glom().item()
      maxG = G[v1,0].glom()
      print 'maxv1:', v1, 'maxG:', maxG

      v2_mask = ((expr_labels > self.tol) * (expr_alpha > self.tol) + (expr_labels < -self.tol) * (expr_alpha < self.C))     
      min_v2 = expr.argmin(G[v2_mask-True]).glom().item()
      minG = G[min_v2,0].glom()
      #print 'minv2:', min_v2, 'minG:', minG
      
      set_v2 = v2_mask.glom().nonzero()[0]
      #print 'actives:', set_v2.shape[0]
      v2 = -1
      for v in set_v2:
        b = maxG - G[v,0].glom()
        if b > self.tol:
          na = (kernel_results[v1,v1] + kernel_results[v,v] - 2*kernel_results[v1,v]).glom()[0][0]
          if na < self.tol: na = 1e12
          
          obj = -(b*b)/na
          if obj <= minObj and v1 != pv1 or v != pv2:
            v2 = v
            a = na
            minObj = obj
      
      if v2 == -1: break
      if maxG - minG < self.tol: break
      
      print 'opt v1:', v1, 'v2:', v2

      pv1 = v1
      pv2 = v2
    
      y1 = labels[v1,0]
      y2 = labels[v2,0]    
        
      oldA1 = alpha[v1,0]
      oldA2 = alpha[v2,0]
      
      # Calculate new alpha values, to reduce the objective function...
      b = y2*expr.glom(gradient[v2,0]) - y1*expr.glom(gradient[v1,0])
      if y1 != y2:
        a += 4 * kernel_results[v1,v2].glom()
      
      newA1 = oldA1 + y1*b/a
      newA2 = oldA2 - y2*b/a   

      # Correct for alpha being out of range...
      sum = y1*oldA1 + y2*oldA2;
  
      if newA1 < self.tol: newA1 = 0.0
      elif newA1 > self.C: newA1 = self.C
     
      newA2 = y2 * (sum - y1 * newA1) 

      if newA2 < self.tol: newA2 = 0.0;
      elif newA2 > self.C: newA2 = self.C
     
      newA1 = y1 * (sum - y2 * newA2)
  
      # Update the gradient...
      dA1 = newA1 - oldA1
      dA2 = newA2 - oldA2
  
      gradient += expr.multiply(labels, kernel_results[:,v1]) * y1 * dA1 + expr.multiply(labels, kernel_results[:,v2]) * y2 * dA2

      alpha[v1,0] = newA1
      alpha[v2,0] = newA2
 
      #print 'alpha:', alpha.glom().T
      
      it += 1
      #print 'gradient:', gradient.glom().T

    self.w = expr.zeros((D, 1), dtype=np.float64).force()
    for i in xrange(D): 
      self.w[i,0] = expr.reduce(alpha, axis=None, dtype_fn=lambda input: input.dtype,
                                local_reduce_fn=margin_mapper,
                                accumulate_fn=np.add, 
                                fn_kw=dict(label=labels, data=expr.force(data[:,i]))).glom()
    
    self.b = 0.0
    E = (labels - self.margins(data)).force()
    
    minB = -1e100
    maxB = 1e100
    actualB = 0.0
    numActualB = 0
    
    for i in xrange(N):
      ai = alpha[i,0]
      yi = labels[i,0]
      Ei = E[i,0]
      
      if ai < 1e-3:
        if yi < self.tol:
          maxB = min((maxB,Ei))
        else:
          minB = max((minB,Ei))
      elif ai > self.C - 1e-3:
        if yi < self.tol:
          minB = max((minB,Ei))
        else:
          maxB = min((maxB,Ei))
      else:
        numActualB += 1
        actualB += (Ei - actualB) / float(numActualB)
    if numActualB > 0:
      self.b = actualB
    else:
      self.b = 0.5*(minB + maxB)

    self.usew_ = True
    print 'iteration finish:', it
    print 'b:', self.b
    print 'w:', self.w.glom()
    
  def fit(self, data, labels, method='smo_1998'):
    if method == 'smo_1998':
      self.train_smo_1998(data, labels)
    elif method == 'smo_2005':
      self.train_smo_2005(data, labels)
    else:
      assert False, "Invalid training method %s" % method
      