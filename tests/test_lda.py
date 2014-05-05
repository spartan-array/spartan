from spartan import expr, util
from spartan.examples.lda import learn_topics
from spartan.expr.write_array import from_file
import test_common
from test_common import millis
import numpy as np
from datetime import datetime

#@test_common.with_ctx
#def test_pr(ctx):
def benchmark_lda(ctx, timer):
  
  print "#worker:", ctx.num_workers
  NUM_TERMS = 1000
  #NUM_DOCS = 200 * ctx.num_workers
  NUM_DOCS = 100 * 64

  # create data
  # NUM_TERMS = 41807
  # NUM_DOCS = 21578
  # terms_docs_matrix = from_file("/scratch/cq/numpy_dense_matrix", sparse = False, tile_hint = (NUM_TERMS, int((NUM_DOCS + ctx.num_workers - 1) / ctx.num_workers))).force()
  
  terms_docs_matrix = expr.randint(NUM_TERMS, NUM_DOCS, low=0, high=100, tile_hint=(NUM_TERMS, NUM_DOCS/ctx.num_workers)).force()
  
  max_iter = 3
  k_topics = 10
  
  t1 = datetime.now()
  doc_topics, topic_term_count = learn_topics(terms_docs_matrix, k_topics, max_iter=max_iter)
  doc_topics.force()
  topic_term_count.force()
  t2 = datetime.now()
  time_cost = millis(t1,t2)
  util.log_warn('total_time:%s ms, train time per iteration:%s ms', time_cost, time_cost/max_iter)
  
  #print "topic_term:", topic_term_count.glom()
  #print "doc_topic:", doc_topics.glom()
  
if __name__ == '__main__':
  test_common.run(__file__)
