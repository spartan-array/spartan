import os
import sys
import logging
import spartan
from time import time

from spartan import blob_ctx, core, util, config
from spartan.rpc import read, serialize, serialize_to, serialization_buffer, FutureGroup

mapper_fn = None
kw = None
futures = None
returnstr = None
results = {}
#mapper_run_time = 0
#mapper_run_count = 0
#read_run_count = 0
#read_run_time = 0


def init():
  config.LOGGING_CONFIGURED = False
  config.parse(sys.argv)
  sys.path.append('./test')


def init_run_kernel(worker_id, worker_ctx, fn):
  global mapper_fn, kw, results, futures
  global read_run_count, read_run_time
  blob_ctx.set(blob_ctx.BlobCtx(worker_id, None, None, worker_ctx))
  #begin = time()
  #w = serialization_buffer.Reader(fn)
  #mapper_fn, kw = read(w)
  mapper_fn, kw = read(fn)
  #read_run_count += 1
  #if read_run_count > 2:
    #read_run_time += time() - begin
  #if read_run_count == 204:
    #util.log_error("read spent %f %d", read_run_time, len(fn))
  results = {}
  futures = FutureGroup()


def map(tid):
  global mapper_fn, kw, results, futures
  global mapper_run_count, mapper_run_time, read_run_time
  tile_id = core.TileId(*tid)
  #begin = time()
  map_result = mapper_fn(tile_id, None, **kw)
  #mapper_run_count += 1
  #mapper_run_time += time() - begin
  #if mapper_run_count == 3264:
    #util.log_error("mapper_fn spent %f", mapper_run_time)
    #util.log_error("read spent %f", read_run_time)
  results[tile_id] = map_result.result
  if map_result.futures is not None:
    assert isinstance(map_result.futures, list)
    futures.extend(map_result.futures)


def wait():
  global futures
  futures.wait()


def finalize():
  global returnstr, results
  w = serialization_buffer.Writer()
  serialize_to(results, w)
  returnstr = w.getvalue()
  #returnstr = serialize(results)
