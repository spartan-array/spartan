from spartan import array, blob_ctx, core, expr
import numpy as np
from spartan import util
import socket
from .helper import *
import numpy as np

def _similarity_mapper(ex, rating_table, similarity_table, item_norm, step):
  ''' Find all pair similarities between items. 
  Parameters
  ----------
  rating_table : Spartan array of shape (M, N).
                 Records the ratings of each item from each user.

  similarity_table : Spartan array of shape (N, N)
                     Records similarities between each item.

  item_norm : Spartan array of shape(N,)
              The norm values of each item.

  step : Integer.
         How many items need to be fetched for each iteration, now this equals to 
         the columns of tiles.
  '''
  M = rating_table.shape[0]
  N = rating_table.shape[1]

  local_ratings = rating_table.fetch(ex)
  local_item_norm = item_norm[ex.ul[1] : ex.lr[1]]
  local_item_norm = local_item_norm.reshape(local_item_norm.shape[0], 1)

  assert local_ratings.shape[0] == M 
  
  # The start index of the items this worker is responsible for.
  local_start_idx = ex.ul[1]  
  # The start index of the items which will be fetched next.
  fetch_start_idx = 0
  count = 0

  while fetch_start_idx < N: 
    util.log_info("Round : %s on %s", count, socket.gethostname())
    # Maybe last tile of the rating matrix doesn't have enough items.
    if N - fetch_start_idx <= step:
      step = N - fetch_start_idx

    count += 1
    
    with util.TIMER.item_fetching:
      # Fetch the ratings of remote items. The matrix is sparse, so this step
      # will not be very expensive.
      remote_ratings = rating_table[:, fetch_start_idx : fetch_start_idx + step]
      remote_item_norm = item_norm[fetch_start_idx : fetch_start_idx + step]
      remote_item_norm = remote_item_norm.reshape(1, remote_item_norm.shape[0])

    with util.TIMER.calculate_similarities:
      '''
      Calculate the all-paris similarities between local items and remote items.
      local_ratings is a local matrix of shape(M, N1), remote_ratings is a local
      matrix of shape(M, N2).

      We calculate the cosine similarity, which is defined as:

          simi(V1, V2) = dot(V1, V2) / (|| V1 || * || V2 ||)

      For effiency, we calculate this in the way of matrix multiplication.
      
      "local_ratings.T.dot(remote_ratings)" generates a N1 X N2 matrix S.
      S[i, j] equals dot(Vi, Vj).
      
      "local_item_norm.dot(remote_item_norm)" generates a N1 X N2 matrix N.
      N[i, j] equals (|| Vi || * || Vj ||).

      In final step, we divide S by N, which yields all-pairs similarity.
      '''
      similarities = local_ratings.T.dot(remote_ratings)
      similarities = np.array(similarities.todense())
      norms = local_item_norm.dot(remote_item_norm)
      similarities = similarities / norms
      # In case some norms are zero. 
      similarities = np.nan_to_num(similarities) 

    # Update this to global array.
    similarity_table[local_start_idx : local_start_idx + similarities.shape[0], 
                     fetch_start_idx : fetch_start_idx + similarities.shape[1]] = similarities

    # Update fetch_start_idx, fetch next part of table.
    fetch_start_idx += step

  result = core.LocalKernelResult()
  result.result = None
  return result


def _select_most_k_similar_mapper(ex, 
                                  similarity_table, 
                                  top_k_similar_table, 
                                  top_k_similar_indices, 
                                  k):
  ''' Find the top k similar items for each item.
  Parameters
  ----------
  similarity_table : Spartan array of shape (N, N)
                     Represents the all pairs similarities.

  top_k_similar_table: Spartan array of shape (N, k)
                       The top k similarity score for each item.

  top_k_similar_indices: Spartan array of shape (N, k)
                         The indices of top k similar items.

  k : Integer
  '''
  local_similarity_table = similarity_table.fetch(ex)
  local_top_k_values = np.zeros((ex.shape[0], k)) 

  start_idx = ex.ul[0] 
  # Find the k largest value of each row. This function is adapted from 
  # bottlenect.argpartsort.
  sorted_indices = argpartsort(local_similarity_table, k, axis=1)[:, :k]
    
  for i in range(sorted_indices.shape[0]):
    local_top_k_values[i] = local_similarity_table[i, sorted_indices[i]]
  
  top_k_similar_indices[ex.ul[0]:ex.lr[0], :] = sorted_indices
  top_k_similar_table[ex.ul[0]:ex.lr[0], :] = local_top_k_values 

  result = core.LocalKernelResult()
  result.result = None
  return result

class ItemBasedRecommender(object):
  def __init__(self, rating_table, k=10, metric="cosine"):
    '''Based on the user-item ratings, recommend items to user.
    Parameters
    ----------
    rating_table : Spartan array of shape (N_USERS, N_ITEMS). 
        Array which represents the ratings of user(M, N)
        M is number of user, N is number of items. Mi,j means
        the rating of user i on item j.

    k : integer. The number of most similar items for each item needs to be precomputed.
        It must be less or equal than the number of items.
    '''
    assert rating_table.shape[1] >= k,\
           "The number of items must be grater or equal than k!"
    self.rating_table = rating_table
    self.k = k

  def _get_norm_of_each_item(self, rating_table):
    """Get norm of each item vector.
    For each Item, caculate the norm the item vector.
    Parameters
    ----------
    rating_table : Spartan matrix of shape(M, N). 
                   Each column represents the rating of the item.

    Returns
    ---------
    item_norm:  Spartan matrix of shape(N,).
                item_norm[i] equals || rating_table[:,i] || 

    """
    ctx = blob_ctx.get()
    if isinstance(rating_table, array.distarray.DistArray):
      rating_table = expr.lazify(rating_table)
    res = expr.sqrt(expr.sum(rating_table ** 2, axis=0, 
                             tile_hint=(rating_table.shape[1] / ctx.num_workers, )))
    return res.force()

  def precompute(self):
    '''Precompute the most k similar items for each item.

    After this funcion returns. 2 attributes will be created.

    Attributes
    ------
    top_k_similar_table : Numpy array of shape (N, k). 
                          Records the most k similar scores between each items. 
    top_k_similar_indices : Numpy array of shape (N, k).
                            Records the indices of most k similar items for each item.
    '''
    M = self.rating_table.shape[0]
    N = self.rating_table.shape[1]
    if isinstance(self.rating_table, expr.Expr):
      self.rating_table = self.rating_table.force()
    
    assert self.rating_table.tile_shape()[0] == M, \
           "rating table is only allowed to tile by columns!"

    self.similarity_table = expr.zeros(shape=(N, N), 
                                       tile_hint=(self.rating_table.tile_shape()[1], N)).force() 

    self.item_norm = self._get_norm_of_each_item(self.rating_table) 
    
    self.rating_table.foreach_tile(mapper_fn=_similarity_mapper,
                                   kw={'rating_table' : self.rating_table,
                                       'similarity_table' : self.similarity_table,
                                       'item_norm' : self.item_norm,
                                       'step' : self.rating_table.tile_shape()[1]})

    # Release the memory for item_norm
    self.item_norm = None
    k = self.k
    top_k_similar_table = expr.zeros((N, k), 
                                      tile_hint=(self.rating_table.tile_shape()[1], k)).force()

    top_k_similar_indices = expr.zeros((N, k), 
                                        tile_hint=(self.rating_table.tile_shape()[1], k), dtype=np.int).force()
    
    
    # Find top-k similar items for each item.
    # Store the similarity scores into table top_k_similar table.
    # Store the indices of top k items into table top_k_similar_indices.
    self.similarity_table.foreach_tile(mapper_fn=_select_most_k_similar_mapper,
                                       kw={'similarity_table' : self.similarity_table,
                                           'top_k_similar_table' : top_k_similar_table,
                                           'top_k_similar_indices' : top_k_similar_indices,
                                           'k' : k})

    self.top_k_similar_table = top_k_similar_table.glom()
    self.top_k_similar_indices = top_k_similar_indices.glom()
