from spartan import expr, blob_ctx
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
    z.force()
    
    failed_worker_id = 0
    ctx = blob_ctx.get()
    ctx.local_worker.mark_bad_tiles(failed_worker_id)
    
    res = z + z
    print res.glom()
