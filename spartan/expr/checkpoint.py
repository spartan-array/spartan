from .base import Expr, lazify
from ..config import FLAGS
from .. import util
from ..node import node_type
from .fio import save
from ..array.distarray import from_replica
from fio import partial_load, load
from .. import blob_ctx

@node_type
class CheckpointExpr(Expr):
  _members = ['children', 'path', 'mode', 'ready']

  def __str__(self):
    return 'checkpoint(expr_id=%s, path=%s)' % (self.expr_id, self.disk)

  def load_data(self, cached_result):
    util.log_info('expr:%s load_data from checkpoint node', self.expr_id)
    if not self.ready:
      return None
    
    if self.mode == 'disk':
      if cached_result is not None:
        util.log_info('load partial disk data')
        ctx = blob_ctx.get()
        extents = ctx.get_workers_for_reload(cached_result)
        new_blobs = partial_load(extents, "%s" % self.expr_id, path = self.path, iszip = False)
        for ex, blob_id in new_blobs.iteritems():
          cached_result.tiles[ex] = blob_id
          cached_result.bad_tiles.remove(ex)
        return cached_result
      else:
        util.log_info('load whole data from disk')
        cached_result = load("%s" % self.expr_id, path = self.path, iszip = False).evaluate()
        return cached_result
    else: # replica
      return None
     
  def _evaluate(self, ctx, deps): 
    result = deps['children']
    if self.mode == 'disk':
      save(result, "%s" % self.expr_id, path = self.path, iszip = False) 
        
    self.ready = True
    return result

def checkpoint(x, mode='disk'):
  '''
  Make a checkpoint for x

  :param x: `numpy.ndarray` or `Expr`
  :param mode: 'disk' or 'replica'
  :rtype: `Expr`
  '''
  return CheckpointExpr(children=lazify(x), path=FLAGS.checkpoint_path, mode=mode, ready=False)
     
