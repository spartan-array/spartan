import cextent_py_if as extent

def find_overlapping(extents, region):
  '''
  Return the extents that overlap with ``region``.   
  
  :param extents: List of extents to search over.
  :param region: `Extent` to match.
  '''
  for ex in extents:
    overlap = extent.intersection(ex, region)
    if overlap is not None:
      yield (ex, overlap)

def all_nonzero_shape(shape):
  '''
  Check if the shape is valid (all elements are biger than zero). This is equal to
  np.all(shape) but is faster because this API doesn't create a numpy array.
  '''
  for i in shape:
    if i == 0:
      return False
  return True

def find_rect(ravelled_ul, ravelled_lr, shape):
  '''
  Return a new (ravellled_ul, ravelled_lr) to make a rectangle for `shape`.
  If (ravelled_ul, ravelled_lr) already forms a rectangle, just return it.

  :param ravelled_ul:
  :param ravelled_lr:
  '''
  if shape[-1] == 1 or ravelled_ul / shape[-1] == ravelled_lr / shape[-1]:
    rect_ravelled_ul = ravelled_ul
    rect_ravelled_lr = ravelled_lr
  else:
    div = 1
    for i in shape[1:]:
      div = div * i
    rect_ravelled_ul = ravelled_ul - (ravelled_ul % div)
    rect_ravelled_lr = ravelled_lr + (div - ravelled_lr % div) % div - 1

  return (rect_ravelled_ul, rect_ravelled_lr)

def is_complete(shape, slices):
  '''
  Returns true if ``slices`` is a complete covering of shape; that is:

  ::

    array[slices] == array

  :param shape: tuple of int
  :param slices: list/tuple of `slice` objects
  :rtype: boolean
  '''
  if len(shape) != len(slices):
    return False

  for dim,slice in zip(shape, slices):
    if slice.start > 0: return False
    if slice.stop < dim: return False
  return True
