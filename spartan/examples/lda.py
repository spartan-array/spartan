import numpy as np
from numpy import linalg
from spartan import expr, util
from spartan.array import distarray, extent

def _lda_train(term_docs_matrix, topic_term_counts, topic_sums, doc_topics, k_topics, alpha, eta, max_iter_per_doc):
  '''
  Using Collapsed Variational Bayes method (Mahout implementation) to train local LDA model.

  Args:
    term_docs_matrix(numpy.ndarray): the count of each term in each document.
    topic_term_counts(numpy.ndarray): the matrix to save p(topic x | term).
    topic_sums(numpy.ndarray): sum_term(p(topic x | term)) for each topic x
    doc_topics(numpy.ndarray): the matrix to save final document/topic inference.
    alpha(float): parameter of LDA model.
    eta(float): parameter of LDA model.
    max_iter_per_doc(int): the max iterations to train each document.
  '''
  for doc_id in range(term_docs_matrix.shape[1]):
    doc = term_docs_matrix[:, doc_id]
    doc_topic_counts = np.ones(k_topics, dtype=np.float64)/k_topics
    topic_term_model = np.zeros((k_topics, term_docs_matrix.shape[0]), dtype=np.float64)
    
    for i in range(max_iter_per_doc):
      # calc un-normalized p(topic x | term, doc)
      for j in doc.nonzero()[0]:
        topic_term_model[:,j] = (topic_term_counts[:,j] + eta) * (doc_topic_counts + alpha) / (topic_sums + eta * doc.shape[0])
      
      # make sure that each of these is properly normalized by topic: sum_x(p(x|t,d)) = 1
      for j in topic_term_model[0].nonzero()[0]:
        topic_term_model[:,j] /= topic_term_model[:,j].sum()
      #topic_term_model /= topic_term_model.sum(axis=0)
      
      # now multiply, term-by-term, by the document, to get the weighted distribution of
      # term-topic pairs from this document.  
      for j in doc.nonzero()[0]:
        topic_term_model[:,j] *= doc[j]
      
      # now recalculate p(topic|doc) by summing contributions from all of p(topic x | term, doc)
      doc_topic_counts = np.linalg.norm(topic_term_model, 1, axis=1)
          
      # now renormalize so that sum_x(p(x|doc)) = 1
      doc_topic_counts = doc_topic_counts / np.linalg.norm(doc_topic_counts, 1)
    
    # update p(topic x | term)
    topic_term_counts += topic_term_model
    topic_sums += np.linalg.norm(topic_term_model, 1, axis=1)
    if doc_topics is not None: doc_topics[doc_id] = doc_topic_counts
    
  return topic_term_counts, topic_sums

def _lda_mapper(array, ex, k_topics, alpha, eta, max_iter_per_doc, topic_term_counts):
  '''
  Using Collapsed Variational Bayes method (Mahout implementation) to train local LDA model.

  Args:
    array(DistArray): the count of each term in each document.
    ex(Extent): Region being processed.
    k_topics: the number of topics we need to find.
    alpha(float): parameter of LDA model.
    eta(float): parameter of LDA model.
    max_iter_per_doc(int): the max iterations to train each document.
    topic_term_counts(DistArray): the matrix to save p(topic x | term).
  '''
  term_docs_matrix = array.fetch(ex)
  local_topic_term_counts = topic_term_counts[:]
  local_topic_sums = np.linalg.norm(local_topic_term_counts, 1, axis=1)
  
  local_topic_term_counts, local_topic_sums = _lda_train(term_docs_matrix, local_topic_term_counts, local_topic_sums, None, k_topics, alpha, eta, max_iter_per_doc)
  
  yield extent.create((0, 0), (k_topics, array.shape[0]),(k_topics, array.shape[0])), local_topic_term_counts

def _lda_doc_topic_mapper(array, ex, k_topics, alpha, eta, max_iter_per_doc, topic_term_counts):
  '''
  Last iteration that uses Collapsed Variational Bayes method (Mahout implementation) to calculate the final document/topic inference.

  Args:
    array(DistArray): the count of each term in each document.
    ex(Extent): Region being processed.
    k_topics: the number of topics we need to find.
    alpha(float): parameter of LDA model.
    eta(float): parameter of LDA model.
    max_iter_per_doc(int): the max iterations to train each document.
    topic_term_counts(DistArray): the matrix to save p(topic x | term).
  '''
  term_docs_matrix = array.fetch(ex)
  local_topic_term_counts = topic_term_counts[:]
  local_topic_sums = np.linalg.norm(local_topic_term_counts, 1, axis=1)
  
  doc_topics = np.ones((term_docs_matrix.shape[1], k_topics), dtype=np.float64)/k_topics
  
  local_topic_term_counts, local_topic_sums = _lda_train(term_docs_matrix, local_topic_term_counts, local_topic_sums, doc_topics, k_topics, alpha, eta, max_iter_per_doc)
  
  yield extent.create((ex.ul[1], 0), (ex.lr[1], k_topics), (array.shape[1], k_topics)), doc_topics
  
def learn_topics(terms_docs_matrix, k_topics, alpha=0.1, eta=0.1, max_iter=10, max_iter_per_doc=1):
  '''
  Using Collapsed Variational Bayes method (Mahout implementation) to train LDA topic model.

  Args:
    terms_docs_matrix(Expr or DistArray): the count of each term in each document.
    k_topics: the number of topics we need to find.
    alpha(float): parameter of LDA model.
    eta(float): parameter of LDA model.
    max_iter(int):the max iterations to train LDA topic model.
    max_iter_per_doc: the max iterations to train each document.
  '''
  topic_term_counts = expr.randn(k_topics, terms_docs_matrix.shape[0], tile_hint=(k_topics, terms_docs_matrix.shape[0])).force()
  for i in range(max_iter):
    new_topic_term_counts = distarray.create((k_topics, terms_docs_matrix.shape[0]), np.float64, reducer=np.add, tile_hint=(k_topics, terms_docs_matrix.shape[0]))
    expr.shuffle(terms_docs_matrix, _lda_mapper, target=new_topic_term_counts, kw={'k_topics': k_topics, 'alpha': alpha, 'eta':eta, 'max_iter_per_doc': max_iter_per_doc, 'topic_term_counts': topic_term_counts}).force()
    topic_term_counts = new_topic_term_counts
  
  doc_topics = expr.shuffle(terms_docs_matrix, _lda_doc_topic_mapper, kw={'k_topics': k_topics, 'alpha': alpha, 'eta':eta, 'max_iter_per_doc': max_iter_per_doc, 'topic_term_counts': topic_term_counts}).force()
  return topic_term_counts, doc_topics
