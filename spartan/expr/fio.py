'''
File I/O for spartan

These include --

* load/save 

  The format is based on npy format, see
  https://github.com/numpy/numpy/blob/master/doc/neps/npy-format.txt for the detail.

  If the ``prefix" for files is "foo", a file, foo_distarray.spf is used to 
  represent the distarray-wise information including array_shape, tiles. For each 
  tile, spartan uses one format-modifed npy file to represent it, foo_$ul_$lr.spf. 
  If the tile is a sparse tile, one additional npz file, foo_$ul_$lr.npz, is used to 
  represent the data (as coo_matrix). 
  All files are in the ``prefix" directory.

  Spartan adds several extra key in ``dictionary" field of npy to record the information 
  of tile. If the tile is sparse, the ``data" field contains nothing since all data 
  is represented by the other npz file. 

* pickle/unpickle

  Just use cPickle to dump/load to/from files. These functions don't have format 
  compatible issues.
'''

import sys
import numpy as np
import scipy.sparse as sp
import cPickle as cpickle
from struct import unpack, pack
from .base import force
from .base import glom
from .map import map
from .ndarray import ndarray
from .reduce import reduce
from .shuffle import shuffle
import os
import bz2
import ast

def spfn(prefix, ul, lr, suffix, ispickle, iszip, isnp = False):
  fn = prefix + "/" + prefix + "_" + str(ul) + "_" + str(lr)
  if suffix != "":
    fn += "_" + suffix 
  if not isnp:
    fn += "_" + "sp"
    if ispickle:
      fn += "p"
    else:
      fn += "f"
    if iszip:
      fn += "bz2"
  return fn

def _save_reducer(ex, tile, axis, prefix = None, sparse = None, iszip = None):
  tile_dict = {'ul' : ex.ul, 'lr' : ex.lr, 'shape' : tile.shape,
               'dtype' : str(tile.dtype), 
               'type' : "SPARSE" if sparse else "DENSITY"}
  cnt = "\x93NUMPY\x01\x00"   # Magic Number & Version
  # Dictionary
  dict_cnt = str(tile_dict)
  if (len(cnt) + 2 + len(dict_cnt)) % 16 != 0:
    dict_cnt += str((16 - (len(cnt) + 2 + len(dict_cnt)) % 16) * ' ')
  cnt += chr(len(dict_cnt) % 256) + chr(len(dict_cnt) / 256) + dict_cnt

  # Now data
  if iszip:
    fn = spfn(prefix, ex.ul, ex.lr, "", False, True)
    fp = bz2.BZ2File(fn, 'w', compresslevel = 1)
  else:
    fn = spfn(prefix, ex.ul, ex.lr, "", False, False)
    fp = open(fn, 'w')

  fp.write(cnt)
  if sparse:
      tile = tile.tocoo()
      save = np.savez_compressed if iszip else np.savez
      save(spfn(prefix, ex.ul, ex.lr, "", True, True, True), 
           row = tile.row, col = tile.col, data = tile.data, shape = tile.shape)
      ret = 2
  else:
      fp.write(tile.data)
      ret = 1
  fp.close()
  return np.asarray(ret)

def save(array, prefix, iszip = False):
  '''
  Save ``array'' to prefix_xxx.
  
  This expr is not lazy and return value is np.ndarray
  Returns number of saved files (not including _dist.spf)

  :param array: Expr or distarray
  :param prefix: Prefix of all file names
  :param iszip: Zip files or not
  '''
  array = force(array)
  if not os.path.exists(prefix):
    os.makedirs(prefix)

  with open(prefix + "/" + prefix + "_dist.spf", "w") as fp:
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
      
  ret = glom(reduce(array, None,
                    dtype_fn=lambda input: np.int64,
                    local_reduce_fn= _save_reducer,
                    accumulate_fn = np.add,
                    fn_kw={'prefix': prefix, 'sparse': array.sparse, 'iszip': iszip}))
  return ret + 1

def _load_mapper(array, ex, prefix = None, sparse = None, dtype = None, iszip = None):
  if iszip:
    fn = spfn(prefix, ex.ul, ex.lr, "", False, True)
    fp = bz2.BZ2File(fn, 'r')
  else:
    fn = spfn(prefix, ex.ul, ex.lr, "", False, False)
    fp = open(fn, 'r') 

  # In current implementation, these information are redundent
  # we don't use them, but keep themfor the future.
  cnt = fp.read(8)  # Magic number and version
  dlen = fp.read(2) # Length of the dictionary
  dlen = ord(dlen[0]) + ord(dlen[1]) * 256
  dict_cnt = fp.read(dlen)
  dict_cnt = ast.literal_eval(dict_cnt) 

  if sparse:
    fp.close()
    a = np.load(spfn(prefix, ex.ul, ex.lr, '', False, True, True) + '.npz')
    return [(ex, sp.coo_matrix((a['data'], (a['row'], a['col'])), a['shape']))]
  else:
    if iszip:
      data = np.frombuffer(fp.read(), dtype=dtype)
    else:
      data = np.fromfile(fp, dtype=dtype)
    data.shape = ex.shape
    fp.close()
    return [(ex, data)]
    


def load(prefix, iszip = False):
  '''
  Load prefix to a new array.

  This expr is lazy and return expr
  Returns a new array with extents/tiles from prefix

  :param prefix: Prefix of all file names
  :param iszip: Zip all files
  '''
  fn = prefix + "/" + prefix + "_dist.spf"
  if not os.path.exists(fn):
    raise IOException

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

  return shuffle(ndarray(shape, dtype=dtype, tile_hint=tile_hint, 
                  sparse=sparse), fn = _load_mapper, 
                  kw = {'prefix' : prefix, 'sparse' : sparse, 
                        'dtype' : dtype, 'iszip' : iszip})

def _pickle_reducer(ex, tile, axis, prefix = None, sparse = None, iszip = None):
  if iszip :
    fn = spfn(prefix, ex.ul, ex.lr, "", True, True)
    with bz2.BZ2File(fn, 'w', compresslevel = 1) as fp:
      cpickle.dump(tile, fp, -1)
  else :
    fn = spfn(prefix, ex.ul, ex.lr, "", True, False)
    with open(fn, "wb") as fp:
      cpickle.dump(tile, fp, -1)

  return np.asarray(1)

def pickle(array, prefix, iszip=False):
  '''
  Save ``array'' to prefix_xxx. Use cPickle.
  
  This expr is not lazy and return value is np.ndarray
  Returns the number of saved files.

  :param array: Expr or distarray
  :param prefix: Prefix of all file names
  '''
  array = force(array)
  if not os.path.exists(prefix):
    os.makedirs(prefix)

  with open(prefix + "/" + prefix + "_dist.spf", "w") as fp:
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

  ret = glom(reduce(array, None,
                    dtype_fn=lambda input: np.int64,
                    local_reduce_fn= _pickle_reducer,
                    accumulate_fn = np.add,
                    fn_kw={'prefix': prefix, 'iszip' : iszip}))
  return ret + 1

def _unpickle_mapper(array, ex, prefix = None, iszip = None):
  if iszip:
    fn = spfn(prefix, ex.ul, ex.lr, "", True, True)
    with bz2.BZ2File(fn, 'r') as fp:
      data = cpickle.load(fp)
  else:
    fn = spfn(prefix, ex.ul, ex.lr, "", True, False)
    with open(fn, "rb") as fp:
      data = cpickle.load(fp)

  return [(ex, data)]

def unpickle(prefix, iszip = False):
  '''
  Load prefix_xxx to a new array. Use cPickle.

  This expr is lazy and return expr
  Returns a new array with extents/tiles from fn

  :param prefix: Prefix of all file names
  '''

  fn = prefix + "/" + prefix + "_dist.spf"
  if not os.path.exists(fn):
    raise IOException

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

  return shuffle(ndarray(shape, dtype=dtype, tile_hint=tile_hint, 
                  sparse=sparse), fn = _unpickle_mapper, 
                  kw = {'prefix' : prefix, 'iszip' : iszip})

