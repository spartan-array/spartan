'''
File I/O for spartan

These include --

* load/save

  The format is based on npy format, see
  https://github.com/numpy/numpy/blob/master/doc/neps/npy-format.txt for the detail.

  If the ``prefix`` for files is "foo", a file, foo_distarray.spf is used to
  represent the distarray-wise information including array_shape, tiles. For each
  tile, spartan uses one format-modifed npy file to represent it, foo_$ul_$lr.spf.
  If the tile is a sparse tile, one additional npz file, foo_$ul_$lr.npz, is used to
  represent the data (as coo_matrix).
  All files are in the ``prefix`` directory.

  Spartan adds several extra key in ``dictionary`` field of npy to record the information
  of tile. If the tile is sparse, the ``data`` field contains nothing since all data
  is represented by the other npz file.

* pickle/unpickle

  Just use cPickle to dump/load to/from files. These functions don't have format
  compatible issues.
'''

import ast
import os
import sys
import bz2
import cPickle as cpickle
import numpy as np
import scipy.sparse as sp
from struct import unpack, pack

from spartan import util
from .operator.base import glom
from .operator.ndarray import ndarray
from .operator.map import map
from .operator.reduce import reduce
from .operator.shuffle import shuffle
from .. import blob_ctx
from ..array import distarray, tile
from ..util import Assert
from ..core import LocalKernelResult


def save_filename(**kw):
  '''
  Return the path to write a save file to, based on the
  input parameters.
  '''
  #TODO(fegin): change this to use default arguments instead of kw and add comments
  fn = kw['path'] + "/" + kw['prefix'] + "/" + kw['prefix'] + \
      "_" + str(kw['ul']) + "_" + str(kw['lr'])
  if kw['suffix'] != "":
    fn += "_" + kw['suffix']
  if not kw['isnp']:
    fn += "_" + "sp"
    if kw['ispickle']:
      fn += "p"
    else:
      fn += "f"
    if kw['iszip']:
      fn += "bz2"
  return fn


def _save_reducer(ex, tile, axis, path=None, prefix=None, iszip=None):
  if not os.path.exists(path + '/' + prefix):
    os.makedirs(path + '/' + prefix)

  sparse = True if sp.issparse(tile) else False
  tile_dict = {'ul': ex.ul, 'lr': ex.lr, 'shape': tile.shape,
               'dtype': str(tile.dtype),
               'type': "SPARSE" if sparse else "DENSITY"}
  cnt = "\x93NUMPY\x01\x00"   # Magic Number & Version
  # Dictionary
  dict_cnt = str(tile_dict)
  if (len(cnt) + 2 + len(dict_cnt)) % 16 != 0:
    dict_cnt += str((16 - (len(cnt) + 2 + len(dict_cnt)) % 16) * ' ')
  cnt += chr(len(dict_cnt) % 256) + chr(len(dict_cnt) / 256) + dict_cnt

  # Now data
  kw = {'path': path, 'prefix': prefix, 'suffix': '',
        'ul': ex.ul, 'lr': ex.lr, 'ispickle': False, 'isnp': False}
  try:
    if iszip:
      kw['iszip'] = True
      fn = save_filename(**kw)
      fp = bz2.BZ2File(fn, 'w', compresslevel=1)
    else:
      kw['iszip'] = False
      fn = save_filename(**kw)
      fp = open(fn, 'w')

    fp.write(cnt)
    if sparse:
        tile = tile.tocoo()
        save = np.savez_compressed if iszip else np.savez
        kw['isnp'] = True
        save(save_filename(**kw), row=tile.row, col=tile.col,
             data=tile.data, shape=tile.shape)
    else:
        fp.write(tile.data)
    fp.close()
  except Exception as e:
    util.log_error('Save %s tile(%s, %s) failed : %s', prefix, ex.ul, ex.lr, e)
    raise

  return np.asarray(1)


def _save(path, prefix, array, iszip):
  path = path + '/' + prefix
  if not os.path.exists(path):
    os.makedirs(path)

  with open(path + '/' + prefix + "_dist.spf", "w") as fp:
    for dim in array.shape:
      fp.write(str(dim) + " ")
    fp.write("\n")
    for dim in array.tile_shape():
      fp.write(str(dim) + " ")
    fp.write("\n")
    fp.write(str(array.dtype) + "\n")
    if array.sparse:
      fp.write("SPARSE\n")
    else:
      fp.write("DENSITY\n")


def save(array, prefix, path='.', iszip=False):
  '''
  Save ``array`` to prefix_xxx.

  This expr is not lazy and return True if success.
  Returns number of saved files (not including _dist.spf)

  :param path: Path to store the directory ``prefix``
  :param array: Expr or distarray
  :param prefix: Prefix of all file names
  :param iszip: Zip files or not

  '''
  array = array.evaluate()
  _save(path, prefix, array, iszip)

  ret = glom(reduce(array, None,
                    dtype_fn=lambda input: np.int64,
                    local_reduce_fn=_save_reducer,
                    accumulate_fn=np.multiply,
                    fn_kw={'prefix': prefix, 'path': path, 'iszip': iszip}))

  return True if ret == 1 else False


def _load_mapper(array, ex, prefix=None, path=None, sparse=None, dtype=None, iszip=None):
  kw = {'path': path, 'prefix': prefix, 'suffix': '',
        'ul': ex.ul, 'lr': ex.lr, 'ispickle': False, 'isnp': False}
  if iszip:
    kw['iszip'] = True
    fn = save_filename(**kw)
    fp = bz2.BZ2File(fn, 'r')
  else:
    kw['iszip'] = False
    fn = save_filename(**kw)
    fp = open(fn, 'r')

  # In current implementation, these information are redundent
  # we don't use them, but keep themfor the future.
  cnt = fp.read(8)  # Magic number and version
  dlen = fp.read(2)  # Length of the dictionary
  dlen = ord(dlen[0]) + ord(dlen[1]) * 256
  dict_cnt = fp.read(dlen)
  dict_cnt = ast.literal_eval(dict_cnt)

  if sparse:
    fp.close()
    kw['isnp'] = True
    a = np.load(save_filename(**kw) + '.npz')
    return [(ex, sp.coo_matrix((a['data'], (a['row'], a['col'])), a['shape']))]
  else:
    if iszip:
      data = np.frombuffer(fp.read(), dtype=dtype)
    else:
      data = np.fromfile(fp, dtype=dtype)
    data.shape = ex.shape
    fp.close()
    return [(ex, data)]


def _load(path, prefix, iszip):
  fn = path + "/" + prefix + "/" + prefix + "_dist.spf"
  if not os.path.exists(fn):
    raise IOError

  with open(fn) as fp:
    line = fp.readline()
    shape = line.strip().split()
    shape = [int(i) for i in shape]
    line = fp.readline()
    tile_hint = line.strip().split()
    tile_hint = [int(i) for i in tile_hint]
    line = fp.readline()
    dtype = np.dtype("".join(line.strip()))
    line = fp.readline()
    sparse = True if line.find("SPARSE") != -1 else False

  return {'shape': shape, 'sparse': sparse, 'dtype': dtype, 'tile_hint': tile_hint}


def load(prefix, path='.', iszip=False):
  '''
  Load prefix to a new array.

  This expr is lazy and return expr
  Returns a new array with extents/tiles from prefix

  :param prefix: Prefix of all file names
  :param path: Path to store the directory ``prefix``
  :param iszip: Zip all files
  '''
  info = _load(path, prefix, iszip)

  return shuffle(ndarray(info['shape'], dtype=info['dtype'],
                         tile_hint=info['tile_hint'], sparse=info['sparse']),
                 fn=_load_mapper,
                 kw={'path': path, 'prefix': prefix, 'sparse': info['sparse'],
                     'dtype': info['dtype'], 'iszip': iszip},
                 shape_hint=info['shape'])


def _pickle_reducer(ex, tile, axis, path=None, prefix=None, sparse=None, iszip=None):
  if not os.path.exists(path + '/' + prefix):
    os.makedirs(path + '/' + prefix)

  kw = {'path': path, 'prefix': prefix, 'suffix': '',
        'ul': ex.ul, 'lr': ex.lr, 'ispickle': True, 'isnp': False}

  try:
    if iszip:
      kw['iszip'] = True
      fn = save_filename(**kw)
      with bz2.BZ2File(fn, 'w', compresslevel=1) as fp:
        cpickle.dump(tile, fp, -1)
    else:
      kw['iszip'] = False
      fn = save_filename(**kw)
      with open(fn, "wb") as fp:
        cpickle.dump(tile, fp, -1)
  except Exception as e:
    util.log_error('Save %s tile(%s, %s) failed : %s' % prefix, ex.ul, ex.lr, e)
    raise

  return np.asarray(1)


def pickle(array, prefix, path='.', iszip=False):
  '''
  Save ``array`` to prefix_xxx. Use cPickle.

  This expr is not lazy and return True if success.
  Returns the number of saved files.

  :param array: Expr or distarray
  :param path: Path to store the directory ``prefix``
  :param prefix: Prefix of all file names
  :param iszip: Zip all files

  '''
  array = array.evaluate()
  _save(path, prefix, array, iszip)

  ret = glom(reduce(array, None,
                    dtype_fn=lambda input: np.int64,
                    local_reduce_fn=_pickle_reducer,
                    accumulate_fn=np.multiply,
                    fn_kw={'path': path, 'prefix': prefix, 'iszip': iszip}))

  return True if ret == 1 else False


def _unpickle_mapper(array, ex, path=None, prefix=None, iszip=None):
  kw = {'path': path, 'prefix': prefix, 'suffix': '',
        'ul': ex.ul, 'lr': ex.lr, 'ispickle': True, 'isnp': False}
  if iszip:
    kw['iszip'] = True
    fn = save_filename(**kw)
    with bz2.BZ2File(fn, 'r') as fp:
      data = cpickle.load(fp)
  else:
    kw['iszip'] = False
    fn = save_filename(**kw)
    with open(fn, "rb") as fp:
      data = cpickle.load(fp)

  return [(ex, data)]


def unpickle(prefix, path='.', iszip=False):
  '''
  Load prefix_xxx to a new array. Use cPickle.

  This expr is lazy and return expr
  Returns a new array with extents/tiles from fn

  :param prefix: Prefix of all file names
  :param path: Path to store the directory ``prefix``
  :param iszip: Zip all files

  '''
  info = _load(path, prefix, iszip)

  return shuffle(ndarray(info['shape'], dtype=info['dtype'],
                         tile_hint=info['tile_hint'], sparse=info['sparse']),
                 fn=_unpickle_mapper,
                 kw={'path': path, 'prefix': prefix, 'iszip': iszip},
                 shape_hint=info['shape'])


def _tile_mapper(tile_id, blob, tiles=None, user_fn=None, **kw):
  for k, v in tiles.iteritems():
    if v == tile_id:
      ex = k
      break

  ctx = blob_ctx.get()
  results = []

  user_result = user_fn(None, ex, **kw)
  if user_result is not None:
    for ex, v in user_result:
      Assert.eq(ex.shape, v.shape, 'Bad shape from %s' % user_fn)
      result_tile = tile.from_data(v)
      tile_id = ctx.create(result_tile).wait().tile_id
      results.append((ex, tile_id))

  return LocalKernelResult(result=results)


def _map_tiles(mapper_fn, tiles, kw=None):
    ctx = blob_ctx.get()

    if kw is None: kw = {}
    kw['tiles'] = tiles
    kw['user_fn'] = mapper_fn

    return ctx.map(tiles.values(), mapper_fn=_tile_mapper, kw=kw)


def _partial_load(path, prefix, extents, iszip, ispickle):
  info = _load(path, prefix, iszip)
  tile_type = tile.TYPE_SPARSE if info['sparse'] else tile.TYPE_DENSE

  ctx = blob_ctx.get()
  tiles = {}
  for ex, i in extents.iteritems():
    tiles[ex] = ctx.create(tile.from_shape(ex.shape, info['dtype'],
                                           tile_type=tile_type),
                           hint=i)

  for ex in extents:
    tiles[ex] = tiles[ex].wait().tile_id

  mapper = _load_mapper if not ispickle else _unpickle_mapper
  if ispickle:
    kw = {'path': path, 'prefix': prefix, 'iszip': iszip}
  else:
    kw = {'path': path, 'prefix': prefix, 'sparse': info['sparse'],
          'dtype': info['dtype'], 'iszip': iszip}
  result = _map_tiles(mapper, tiles, kw=kw)

  loaded_tiles = {}
  for tile_id, v in result.iteritems():
    for ex, newid in v:
      loaded_tiles[ex] = newid

  distarray._pending_destructors.extend(tiles.values())

  return loaded_tiles


def partial_load(extents, prefix, path=".", iszip=False):
  '''
  Load some tiles from ``prefix`` to some workers.

  This expr is not lazy and return tile_id(s).

  :param extents: A dictionary which contains extents->workers
  :param prefix: Prefix of all file names
  :param path: Path to store the directory ``prefix``
  :param iszip: Zip all files
  :rtype: A dictionary which contains extents->tile_id

  '''
  return _partial_load(path, prefix, extents, iszip, False)


def partial_unpickle(extents, prefix, path=".", iszip=False):
  '''
  Unpickle some tiles from ``prefix`` to some workers.

  This expr is not lazy and return tile_id(s).

  :param extents: A dictionary which contains extents->workers
  :param prefix: Prefix of all file names
  :param path: Path to store the directory ``prefix``
  :param iszip: Zip all files
  :rtype: A dictionary which contains extents->tile_ids

  '''
  return _partial_load(path, prefix, extents, iszip, True)
