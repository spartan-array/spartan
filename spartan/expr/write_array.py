from .base import Expr
from .ndarray import ndarray
from ..node import Node, node_type
from spartan.array import tile, distarray, extent
import numpy as np

@node_type
class WriteArrayExpr(Expr):
  _members = ['array', 'npa']

  def __str__(self):
    return 'dist_array(%s, %s)' % (self.npa.shape, self.npa.dtype)
  
  def visit(self, visitor):
    return WriteArrayExpr(array = self.array, npa = self.npa)
  
  def dependencies(self):
    return dict([(k, getattr(self, k)) for k in self.members])
  
  def compute_shape(self):
    return self.npa.shape;
 
  def _evaluate(self, ctx, deps):
    shape = deps['npa'].shape
    lr = shape
    ul = [0 * i for i in shape]
    ex = extent.create(ul, lr, shape)
    deps['array'].update(ex, deps['npa'])
    return deps['array']

def make_from_numpy(source):
  '''
  Make a distarray from a numpy array

  :param source: `numpy.ndarray` or npy/npz file name 
  :rtype: `Expr`
  '''
  if isinstance(source, str):
    npa = np.load(source)
    if source.endswith("npz"):
      # We expect only one npy in npz
      for k, v in npa.iteritems():
        source = v
      npa.close()
      npa = source
  elif isinstance(source, np.ndarray):
    npa = source
  else:
    raise TypeError
  
  array = ndarray(shape = npa.shape, dtype = npa.dtype)
  return WriteArrayExpr(array = array, npa = npa)
     
