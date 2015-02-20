'''
Operations for updating slices of arrays.

To preserve the non-mutation semantics required for optimizations
to be correct, writing to an array should not actually mutate the
original array, but should instead create a new array with the
appropriate region updated.  This code currently mutates arrays
in place, and therefore should be used with care.
'''

import numpy as np
import scipy.sparse as sp
import scipy
import os
import math
import ast
import struct

from traits.api import PythonValue

from spartan import rpc
from spartan import master
from spartan.array import distarray, extent, sparse
from .base import Expr
from .ndarray import ndarray
from .shuffle import shuffle
from .slice import Slice
from ...core import LocalKernelResult
from ...util import Assert, FileHelper


def _write_mapper(ex, source=None, sregion=None, dst_slice=None):
  intersection = extent.intersection(ex, sregion)

  futures = rpc.FutureGroup()
  if intersection is not None:
    dst_lr = np.asarray(intersection.lr) - np.asarray(sregion.ul)
    dst_ul = np.asarray(intersection.ul) - np.asarray(sregion.ul)
    dst_ex = extent.create(tuple(dst_ul), tuple(dst_lr), dst_slice.shape)
    v = dst_slice.fetch(dst_ex)
    futures.append(source.update(intersection, v, wait=False))

  return LocalKernelResult(result=None, futures=futures)


class WriteArrayExpr(Expr):
  array = PythonValue(None, desc="DistArray or Expr")
  src_slices = PythonValue(None, desc="Slices or a tuple of slices")
  data = PythonValue(None, desc="np.ndarray or Expr")
  dst_slices = PythonValue(None, desc="Slices or a tuple of slices")

  def __str__(self):
    return 'WriteArrayExpr[%d] %s %s' % (self.expr_id, self.array, self.data)

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
      dst_slice = Slice(data, dst_slices)
      Assert.eq(sregion.shape, dst_slice.shape)
      array.foreach_tile(mapper_fn=_write_mapper,
                         kw={'source': array, 'sregion': sregion,
                             'dst_slice': dst_slice})
    else:
      raise TypeError

    return array

  def compute_shape(self):
    return self.array.shape


def write(array, src_slices, data, dst_slices):
  '''
  array[src_slices] = data[dst_slices]

  :param array: Expr or distarray
  :param src_slices: slices for array
  :param data: data
  :param dst_slices: slices for data
  :rtype: `Expr`

  '''
  return WriteArrayExpr(array=array, src_slices=src_slices,
                        data=data, dst_slices=dst_slices)


def _local_load_reducer(old, new):
  return new + old

#def _local_read_dense_mm(ex, fn, data_begin, data_size):


def _local_read_sparse_mm(array, ex, fn, data_begin):
  '''
  1. Noted that Matrix Market format doesn't require (row, col) to be sorted.
     If the file is sorted (by either row or col), each worker will return
     only a part of the array. If the file is unsorted, each worker may
     return a very big and sparser sub-array of the original array. In the
     worst case, the sub-array can be as large as the original array but
     sparser.
  2. We can't know how many lines without reading the whole file. So we simply
     decide the region this worker should read based on the file size.
  '''
  data_size = os.path.getsize(fn) - data_begin
  array_size = np.product(array.shape)
  begin = extent.ravelled_pos(ex.ul, array.shape)
  begin = math.ceil(((begin * 1.0) / array_size) * data_size) + data_begin
  end = extent.ravelled_pos([(i - 1) for i in ex.lr], array.shape)
  end = math.floor(((end * 1.0) / array_size) * data_size) + data_begin

  ul = [array.shape[0], array.shape[1]]
  lr = [0, 0]
  rows = []
  cols = []
  data = []
  with open(fn) as fp:
    fp.seek(begin)
    if begin != data_begin:
      fp.seek(begin - 1)
      a = fp.read(1)
      if a != '\n':
        line = fp.readline()

    pos = fp.tell()
    for line in fp:
      if pos > end + 1:  # +1 in case end locates on \n
        break
      pos += len(line)
      (_row, _col), val = _extract_mm_coordinate(line)
      _row -= 1
      _col -= 1
      rows.append(_row)
      cols.append(_col)
      data.append(float(val))
      ul[0] = _row if _row < ul[0] else ul[0]
      ul[1] = _col if _col < ul[1] else ul[1]
      lr[0] = _row if _row > lr[0] else lr[0]
      lr[1] = _col if _col > lr[1] else lr[1]

  # Adjust rows and cols based on the ul of this submatrix.
  for i in xrange(len(rows)):
    rows[i] -= ul[0]
    cols[i] -= ul[1]

  new_ex = extent.create(ul, [lr[0] + 1, lr[1] + 1], array.shape)
  new_array = sp.coo_matrix((data, (rows, cols)), new_ex.shape)
  return new_ex, sparse.convert_sparse_array(new_array)


def _readmm_mapper(array, ex, fn=None, data_begin=None):
  if array.sparse:
    new_ex, new_array = _local_read_sparse_mm(array, ex, fn, data_begin)
  else:
    pass

  yield new_ex, new_array


def _extract_mm_coordinate(shape_info):
  shape_info = shape_info.split()
  shape = [int(i) for i in shape_info[:-1]]
  edges = shape_info[-1]
  return (shape, edges)


def _parse_mm_header(fn, sparse_threshold=0.01):
  with open(fn) as fp:
    line = fp.readline().strip()
    header = line
    while line[0] == '%':
      line = fp.readline()
    shape_info = line
    data_begin = fp.tell()

  if header.find('real'):
    dtype = np.float
  elif header.find('integer'):
    dtype = np.int

  shape, edges = _extract_mm_coordinate(shape_info)
  edges = int(edges)
  if (edges * 1.0) / np.product(shape) < sparse_threshold:
    sparse = True
  else:
    sparse = False

  return (shape, dtype, sparse, data_begin)


def _bulk_read(fp, size, bulk_size=2**27):
  '''
  size must be 4, 8, 16 or 32. build_size must be 2^n
  '''
  assert(size == 4 or size == 8 or size == 16 or size == 32)
  while True:
    data = fp.read(bulk_size)
    if not data:
      break
    tell = 0
    data_len = len(data)
    while tell + size <= data_len:
      tell += size
      yield data[(tell - size):tell]


def _local_read_sparse_npy(array, ex, fn):
  '''
  1. Noted that coo_matrix format doesn't require row[] or col[] to be sorted.
     If one of row[] or col[] is sorted (by either row or col), each worker will
     return only a part of the array. If the file is unsorted, each worker may
     return a very big and sparser sub-array of the original array. In the worst
     case, the sub-array can be as large as the original array but sparser.
  2. For numpy format, we can evenly distribute the files we need to read to
     workers.
  '''
  #data_begin = {}
  #dtype = {}
  #dtype_size = {}
  #shape = {}
  #fp = {}
  #read_next = {}
  attr = {'data_begin': {}, 'dtype': {}, 'shape': None,
          'read_next': {}, 'fn': {}}
  types = ['row', 'col', 'data']
  dtype_name = {'float64': 'd', 'float32': 'f', 'int64': 'q', 'int32': 'i'}

  for i in types:
    _fn = '%s_%s.npy' % (fn, i)
    attr['fn'][i] = _fn
    _shape, attr['dtype'][i], attr['data_begin'][i] = _parse_npy_header(_fn)
    if attr['shape'] is not None:
      assert attr['shape'] == _shape
    else:
      attr['shape'] = _shape
  #shape['row'], dtype['row'], data_begin['row'] = _parse_npy_header(fn + '_row.npy')
  #shape['col'], dtype['col'], data_begin['col'] = _parse_npy_header(fn + '_col.npy')
  #shape['data'], dtype['data'], data_begin['data'] = _parse_npy_header(fn + '_data.npy')

  item_count = np.product(array.shape)
  begin_item = extent.ravelled_pos(ex.ul, array.shape)
  begin_item = int(math.ceil(((begin_item * 1.0) / item_count) * attr['shape'][0]))
  end_item = extent.ravelled_pos([(i - 1) for i in ex.lr], array.shape)
  end_item = int(math.floor((end_item * 1.0) / item_count * attr['shape'][0])) + 1
  end_item = attr['shape'][0] if end_item > attr['shape'][0] else end_item

  ul = [array.shape[0], array.shape[1]]
  lr = [0, 0]
  rows = []
  cols = []
  data = []
  with FileHelper(row=open(attr['fn']['row'], 'rb'),
                  col=open(attr['fn']['col'], 'rb'),
                  data=open(attr['fn']['data'], 'rb')) as fp:
    for k in types:
      _dtype = attr['dtype'][k]
      _dtype_size = _dtype.itemsize
      _fp = getattr(fp, k)

      _fp.seek(attr['data_begin'][k] + begin_item * _dtype_size)
      attr['read_next'][k] = _bulk_read(_fp, _dtype_size)
      attr['dtype'][k] = dtype_name[_dtype.name]

    for i in xrange(begin_item, end_item):
      _row = struct.unpack(attr['dtype']['row'], attr['read_next']['row'].next())[0]
      rows.append(_row)
      _col = struct.unpack(attr['dtype']['col'], attr['read_next']['col'].next())[0]
      cols.append(_col)
      _data = struct.unpack(attr['dtype']['data'], attr['read_next']['data'].next())[0]
      data.append(_data)

      ul[0] = _row if _row < ul[0] else ul[0]
      ul[1] = _col if _col < ul[1] else ul[1]
      lr[0] = _row if _row > lr[0] else lr[0]
      lr[1] = _col if _col > lr[1] else lr[1]

  for i in xrange(len(rows)):
    rows[i] -= ul[0]
    cols[i] -= ul[1]

  new_ex = extent.create(ul, [lr[0] + 1, lr[1] + 1], array.shape)
  new_array = sp.coo_matrix((data, (rows, cols)), new_ex.shape)
  return new_ex, sparse.convert_sparse_array(new_array)


def _readnpy_mapper(array, ex, fn=None):
  if array.sparse:
    new_ex, new_array = _local_read_sparse_npy(array, ex, fn)
  else:
    pass

  yield (new_ex, new_array)


def _parse_npy_header(fn):
  with open(fn) as fp:
    fp.seek(8)  # Skip magic
    dict_len = ord(fp.read(1)) + ord(fp.read(1)) * 256
    dict_str = fp.read(dict_len)
    dict_cnt = ast.literal_eval(dict_str)
    data_begin = fp.tell()

  return (dict_cnt['shape'], np.dtype(dict_cnt['descr']), data_begin)


def from_file_parallel(fn, file_format='mm', sparse=True, tile_hint=None):
  '''
  Make a distarray from a file or files. The file(s) will be read by workers.
  Therefore, the file(s) should be located in a shared file system such as HDFS.

  This API currently supports:
    numpy(sparse)
    Matrix Market format(sparse).

  Matrix Market:
    http://math.nist.gov/MatrixMarket/formats.html
    Sample Python code to save a coo_matrix to a Matrix Market format file:
      scipy.io.mmwrite('xxx.mtx', coo)

  Numpy:
    Since numpy format doesn't actually support sparse arrays, we assume that
    there are four npy files if the file format is numpy. These files store row,
    col, data and shape for a coo_matrix. Their file name should be fn_row.npy,
    fn_col.npy, fn_data.npy and fn_shape.npy where fn is the first argument of
    this API. For efficient loading, the row and column matrices should be
    sorted. Unsorted matrices are supported, but performance will be poor.
    Sample python code to save a coo_matirx:
      numpy.save('xxx_row.npy', coo.row)
      numpy.save('xxx_col.npy', coo.col)
      numpy.save('xxx_data.npy', astype(coo.data.astype(numpy.float32)))
      numpy.save('xxx_shape.npy', numpy.asarray(coo.shape, dtype = numpy.float32))

  Args
    fn: `file name`
    file_format: `The format of fn`
    sparse: `Sparse array or not`
    tile_hint: `tile hint`
  Return
    Expr
  '''

  if file_format == 'numpy':
    shape = list(np.load(fn + '_shape.npy'))
    if sparse:
      mapper = _readnpy_mapper
      reducer = _local_load_reducer
      _shape, dtype, _data_begin = _parse_npy_header(fn + '_data.npy')
      kw = {'fn': fn}
    else:
      raise NotImplementedError("Only support sparse numpy now.")
  elif file_format == 'mm':
    shape, dtype, sparse, data_begin = _parse_mm_header(fn)
    if not sparse:
      raise NotImplementedError("Only support sparse mm now.")
    if len(shape) != 2:
      raise NotImplementedError("Only support two-dimension sparse mm now.")
    mapper = _readmm_mapper
    reducer = _local_load_reducer
    kw = {'fn': fn, 'data_begin': data_begin}
  else:
    raise NotImplementedError("Only support mm now. Got %s" % file_format)

  array_tile_hint = distarray.good_tile_shape(shape, num_shards=master.get().num_workers)

  array = ndarray(shape=shape, dtype=dtype, sparse=sparse, tile_hint=array_tile_hint)
  target = ndarray(shape=shape, dtype=dtype, sparse=sparse,
                   tile_hint=tile_hint, reduce_fn=reducer)
  return shuffle(array, fn=mapper, kw=kw, target=target)


def from_file(fn, file_type='numpy', sparse=True, tile_hint=None):
  '''
  Make a distarray from a file.

  This API Currently supports:
    numpy(dense/sparse)
    Matrix Market format(dense/sparse).

  The detail file format descriptions are in the comment in from_file_parallel().

  Args
    fn: `file name`
    file_format: `The format of fn`
    sparse: `Sparse array or not`
    tile_hint: `tile hint`
  Return
    Expr
  '''

  if file_type == 'numpy':
    if sparse:
      shape = list(np.load(fn + '_shape.npy'))
      row = np.load(fn + '_row.npy')
      col = np.load(fn + '_col.npy')
      data = np.load(fn + '_data.npy')
      npa = sp.coo_matrix((data, (row, col)), shape=shape)
    else:
      npa = np.load(fn)
      if fn.endswith("npz"):
        # We expect only one npy in npz
        for k, v in npa.iteritems():
          fn = v
        npa.close()
        npa = fn
  elif file_type == 'mm':
    npa = scipy.io.mmread(fn)
    if sp.issparse(npa) and npa.dtype == np.float64:
      npa = npa.astype(np.float32)
  else:
    raise NotImplementedError("Only support npy and mm now. Got %s" % file_type)

  return from_numpy(npa, tile_hint)


def from_numpy(npa, tile_hint=None):
  '''
  Make a distarray from a numpy array

  Args
    npa: `numpy.ndarray`
    tile_hint: `tile hint`
  Return
    Expr
  '''
  if (not isinstance(npa, np.ndarray)) and (not sp.issparse(npa)):
    raise TypeError("Expected ndarray, got: %s" % type(npa))

  # if the sparse type can't support slice, we need to convert it to another type.
  if sp.issparse(npa):
    npa = npa.tocsr()

  array = ndarray(shape=npa.shape, dtype=npa.dtype,
                  sparse=sp.issparse(npa), tile_hint=tile_hint)
  slices = tuple([slice(0, i) for i in npa.shape])

  return write(array, slices, npa, slices)
