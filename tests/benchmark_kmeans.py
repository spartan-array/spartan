import test_common
from spartan.examples import kmeans

  
def benchmark_kmeans(ctx, timer):
  num_pts = ctx.num_workers() * 1000000
  num_dim = 10
  num_centers = 1000
  
  timer.time_op('k-means', lambda: kmeans.run(num_pts, num_centers, num_dim))
  timer.time_op('k-means', lambda: kmeans.run(num_pts, num_centers, num_dim))
  timer.time_op('k-means', lambda: kmeans.run(num_pts, num_centers, num_dim))
  
if __name__ == '__main__':
  test_common.run(__file__)
