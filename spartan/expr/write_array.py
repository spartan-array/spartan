'''
Distarray write operations and expr.
'''

import numpy as np
from spartan import rpc
from .base import Expr
from .ndarray import ndarray
from ..node import Node, node_type
from spartan.array import tile, distarray, extent
from .. import util
from ..util import Assert
from.map import MapResult

def _write_mapper(ex, source = None, sregion = None, dslice = None):
  intersection = extent.intersection(ex, sregion)

  futures = rpc.FutureGroup()
  if intersection != None:
    dlr = np.asarray(intersection.lr) - np.asarray(sregion.ul)
    dul = np.asarray(intersection.ul) - np.asarray(sregion.ul)
    dex = extent.create(tuple(dul), tuple(dlr), dslice.shape)
    v = dslice.fetch(dex)
    futures.append(source.update(intersection, v, wait=False))

  return MapResult(None, futures)


@node_type
class WriteArrayExpr(Expr):
  _members = ['array', 'sslices', 'data', 'dslices']

  def __str__(self):
    return 'WriteArrayExpr[%d] %s %s %s' % (self.expr_id, self.array, self.data)
  
  def _evaluate(self, ctx, deps):
    array = deps['array']
    sslices = deps['sslices']
    data = deps['data']
    dslices = deps['dslices']

    sregion = extent.from_slice(sslices, array.shape)
    if isinstance(data, np.ndarray):
      if sregion.shape == data.shape:
         array.update(sregion, data)
      else:
         array.update(sregion, data[dslices])
    elif isinstance(data, distarray.DistArray):
      dslice = distarray.Slice(data, dslices)
      Assert.eq(sregion.shape, dslice.shape)
      array.foreach_tile(mapper_fn = _write_mapper,
                         kw = {'source':array, 'sregion':sregion,
                               'dslice':dslice})
    else:
      raise TypeError

    return array


def write(array, sslices, data, dslices):
  '''
  array[sslices] = data[dslices]

  :param array: Expr or distarray
  :param sslices: slices for array
  :param data: data
  :param dslices: slices for data
  :rtype: `Expr`
  '''
  return WriteArrayExpr(array = array, sslices = sslices,
                        data = data, dslices = dslices)


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
  slices = tuple([slice(0, i) for i in npa.shape])

  return write(array, slices, npa, slices)
     
