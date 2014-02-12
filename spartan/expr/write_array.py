'''
Distarray write operations and expr.
'''

import numpy as np
import scipy.sparse as sp
from spartan import rpc
from .base import Expr
from .ndarray import ndarray
from ..node import Node, node_type
from spartan.array import tile, distarray, extent
from .. import util
from ..util import Assert
from.map import MapResult

def _write_mapper(ex, source = None, sregion = None, dst_slice = None):
  intersection = extent.intersection(ex, sregion)

  futures = rpc.FutureGroup()
  if intersection != None:
    dst_lr = np.asarray(intersection.lr) - np.asarray(sregion.ul)
    dst_ul = np.asarray(intersection.ul) - np.asarray(sregion.ul)
    dst_ex = extent.create(tuple(dst_ul), tuple(dst_lr), dst_slice.shape)
    v = dst_slice.fetch(dst_ex)
    futures.append(source.update(intersection, v, wait=False))

  return MapResult(None, futures)


@node_type
class WriteArrayExpr(Expr):
  _members = ['array', 'src_slices', 'data', 'dst_slices']

  def __str__(self):
    return 'WriteArrayExpr[%d] %s %s %s' % (self.expr_id, self.array, self.data)
  
  def _evaluate(self, ctx, deps):
    array = deps['array']
    src_slices = deps['src_slices']
    data = deps['data']
    dst_slices = deps['dst_slices']

    sregion = extent.from_slice(src_slices, array.shape)
    if isinstance(data, np.ndarray) or sp.issparse(data):
      if sregion.shape == data.shape:
         array.update(sregion, data)
      else:
         array.update(sregion, data[dst_slices])
    elif isinstance(data, distarray.DistArray):
      dst_slice = distarray.Slice(data, dst_slices)
      Assert.eq(sregion.shape, dst_slice.shape)
      array.foreach_tile(mapper_fn = _write_mapper,
                         kw = {'source':array, 'sregion':sregion,
                               'dst_slice':dst_slice})
    else:
      raise TypeError

    return array


def write(array, src_slices, data, dst_slices):
  '''
  array[src_slices] = data[dst_slices]

  :param array: Expr or distarray
  :param src_slices: slices for array
  :param data: data
  :param dst_slices: slices for data
  :rtype: `Expr`
  '''
  return WriteArrayExpr(array = array, src_slices = src_slices,
                        data = data, dst_slices = dst_slices)

# TODO: Many applications use matlab format. Maybe we should support it.
def from_file(fn, file_type = 'numpy'):
  '''
  Make a distarray from a file.
  Currently support npy/npz.

  :param fn: `file name`
  :rtype: `Expr`
  '''

  if file_type == 'numpy':
    npa = np.load(fn)
    if fn.endswith("npz"):
      # We expect only one npy in npz
      for k, v in npa.iteritems():
        fn = v
      npa.close()
      npa = fn
  else:
    raise NotImplementedError("Only support npy/npz now. Got %s" % file_type)

  return from_numpy(npa)

def from_numpy(npa):
  '''
  Make a distarray from a numpy array

  :param npa: `numpy.ndarray`
  :rtype: `Expr`
  '''
  if (not isinstance(npa, np.ndarray)) and (not sp.issparse(npa)):
    raise TypeError("Expected ndarray, got: %s" % type(npa))
  
  array = ndarray(shape = npa.shape, dtype = npa.dtype, sparse = sp.issparse(npa))
  slices = tuple([slice(0, i) for i in npa.shape])

  return write(array, slices, npa, slices)
     
