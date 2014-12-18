import os
import logging
import sys
from time import time

from spartan import blob_ctx, core, util, config
from spartan.rpc import read, serialize, serialization_buffer, FutureGroup

mapper_fn = None
kw = None
futures = None
returnstr = None
results = {}
mapper_run_time = 0
mapper_run_count = 0

def init(worker_id, worker_ctx, fn):
  global mapper_fn, kw, results, futures
  blob_ctx.set(blob_ctx.BlobCtx(worker_id, None, None, worker_ctx))
  begin = time()
  #reader = serialization_buffer.Reader(fn)
  #mapper_fn, kw = read(reader)
  mapper_fn, kw = read(fn)
  #util.log_error("read spent %f %d", time() - begin, len(fn))
  results = {}
  futures = FutureGroup()


def map(tid):
  global mapper_fn, kw, results, futures, mapper_run_time, mapper_run_count
  tile_id = core.TileId(*tid)
  begin = time()
  map_result = mapper_fn(tile_id, None, **kw)
  mapper_run_count += 1
  mapper_run_time += time() - begin
  if mapper_run_count == 3264:
    util.log_error("mapper_fn spent %f", mapper_run_time)
  results[tile_id] = map_result.result
  if map_result.futures is not None:
    assert isinstance(map_result.futures, list)
    futures.extend(map_result.futures)


def wait():
  global futures
  futures.wait()


def finalize():
  global returnstr, results
  returnstr = serialize(results)
