"""
Sparse array support.

This does not yet exist, but should eventually contain:

  * Individual sparse tiles
  * `spartan.distarray.DistArray`-like object
  
A normal DistArray *might* work, but this depends on the
distribution and size of the sparse array.

Additionally, it might be nice to support infinite-dimensional "arrays"; 
these correspond to the more traditional key-value stores used in most
distributed systems.  Supporting these requires that we ensure the various
high-level operators are written in a way that isn't implicitly relying
on having a known number of dimensions.  (Unless this is an explicit 
requirement (e.g dot product)).   
"""