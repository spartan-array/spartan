import numpy as np

from ..array import distarray, extent
from .. import util, blob_ctx
from ..util import Assert


def broadcast_mapper(ex, tile, mapper_fn=None, bcast_obj=None):
  raise NotImplementedError

def _broadcast_mapper(tile_id, blob, apply_region, array = None, user_fn = None, **kw):
    base_ex = array.base.extent_for_blob(tile_id)
    ul = [0 for dim in array.shape]
    lr = [dim for dim in array.shape]

    for i in range(len(base_ex.ul) - 1, 0, -1):
      broadcast_i = i + array.prepend_dim
      if array.base.shape[i] != array.shape[broadcast_i]:
        # array.base.shape[i] must be 1 according to the broadcast rules.
        assert(ul[i + array.prepend_dim] == 0)
        lr[i + array.prepend_dim] = array.shape[broadcast_i]
    ex = extent.create(ul, lr, array.shape)

    return user_fn(ex, **kw)

class Broadcast(distarray.DistArray):
  '''Mimics the behavior of Numpy broadcasting.

  Takes an input of shape (x, y) and a desired output shape (x, y, z),
  the broadcast object reports shape=(x,y,z) and overrides __getitem__
  to return the appropriate values.
  '''
  def __init__(self, base, shape):
    Assert.isinstance(base, (np.ndarray, distarray.DistArray))
    Assert.isinstance(shape, tuple)
    self.base = base
    self.shape = shape
    self.dtype = base.dtype
    self.bad_tiles = []
    self.prepend_dim = len(shape) - len(base.shape)


  def __repr__(self):
    return 'Broadcast(%s -> %s)' % (self.base, self.shape)


  def real_size(self):
    '''Return the size of the underlying array.

    Offset by one to prefer direct arrays over broadcasts.
    '''
    return np.prod(self.base.shape) - 1

  def foreach_tile(self, mapper_fn, kw=None):
    ctx = blob_ctx.get()

    if kw is None: kw = {}
    kw['array'] = self
    kw['user_fn'] = mapper_fn

    return ctx.map(self.base.tiles.values(),
                   mapper_fn = _broadcast_mapper,
                   kw=kw)

  def fetch(self, ex):
    # make a template to pass to numpy broadcasting
    template = np.ndarray(ex.shape, dtype=self.base.dtype)

    # convert the extent to the base form

    # first drop extra dimensions
    while len(ex.shape) > len(self.base.shape):
      ex = extent.drop_axis(ex, 0)

    # fold down expanded dimensions
    ul = []
    lr = []
    for i in xrange(len(self.base.shape)):
      size = self.base.shape[i]
      if size == 1:
        ul.append(0)
        lr.append(1)
      else:
        ul.append(ex.ul[i])
        lr.append(ex.lr[i])

    ex = extent.create(ul, lr, self.base.shape)
    fetched = self.base.fetch(ex)

    _, bcast = np.broadcast_arrays(template, fetched)

    util.log_debug('bcast: %s %s', fetched.shape, template.shape)
    return bcast


def broadcast(args):
  '''Convert the list of arrays in ``args`` to have the same shape.

  Extra dimensions are added as necessary, and dimensions of size
  1 are repeated to match the size of other arrays.

  :param args: List of `DistArray`
  '''

  if len(args) == 1:
    return args

  orig_shapes = [list(x.shape) for x in args]
  dims = [len(shape) for shape in orig_shapes]
  max_dim = max(dims)
  new_shapes = []

  # prepend filler dimensions for smaller arrays
  for i in range(len(orig_shapes)):
    diff = max_dim - len(orig_shapes[i])
    new_shapes.append([1] * diff + orig_shapes[i])

  # check shapes are valid
  # for each axis, all arrays should either share the
  # same size, or have size == 1
  for axis in range(max_dim):
    axis_shape = set(shp[axis] for shp in new_shapes)

    assert len(axis_shape) <= 2, 'Mismatched shapes for broadcast: %s' % orig_shapes
    if len(axis_shape) == 2:
      assert 1 in axis_shape, 'Mismatched shapes for broadcast: %s' % orig_shapes

    # now lift the inputs with size(axis) == 1
    # to have the maximum size for the axis
    max_size = max(shp[axis] for shp in new_shapes)
    for shp in new_shapes:
      shp[axis] = max_size

  # wrap arguments with missing dims in a Broadcast object.
  results = []
  for i in range(len(args)):
    if new_shapes[i] == orig_shapes[i]:
      results.append(args[i])
    else:
      results.append(Broadcast(args[i], tuple(new_shapes[i])))

  #util.log_debug('Broadcast result: %s', results)
  return results

