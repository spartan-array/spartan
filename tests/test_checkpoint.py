from spartan import expr, blob_ctx
from spartan.util import Assert
import numpy as np
import test_common


ARRAY_SIZE=(10,10)


class TestCheckpoint(test_common.ClusterTest):
  def test1(self):
    a = expr.ones(ARRAY_SIZE)
    b = expr.ones(ARRAY_SIZE)
    c = expr.ones(ARRAY_SIZE)
    x = a + b + c
    y = x + x
    z = y + y
    z = expr.checkpoint(z, mode='disk')
    z.evaluate()

    failed_worker_id = 0
    ctx = blob_ctx.get()
    ctx.local_worker.mark_failed_worker(failed_worker_id)

    res = z + z
    Assert.all_eq(res.glom(), np.ones(ARRAY_SIZE)*24)
