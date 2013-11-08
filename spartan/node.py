'''Helper for constructing trees of objects.

Provides pretty printing, equality testing, hashing and keyword initialization.
'''  


import numpy as np
from itertools import izip
import copy

class NodeTemplate(object):
  def __init__(self, *args, **kw):
    n_args = len(args)
    members = self.members
    n_members = len(members)
    class_name = self.__class__.__name__
    self_dict = self.__dict__
    if n_args == n_members:
      assert len(kw) == 0
      for (k,v) in zip(members,args):
        self_dict[k] = v
    elif n_args < n_members:
      for field in members:
        self_dict[field] = kw.get(field)

      for field, value in izip(members, args):
        self_dict[field] = value

      for (k,v) in kw.iteritems():
        assert k in members, \
          "Keyword argument '%s' not recognized for %s: %s" % \
          (k, self.node_type(), members)
    else:
      raise Exception('Too many arguments for %s, expected %s' % \
                      (class_name, members))

    # it's more common to not define a node initializer,
    # so add an extra check to avoid having to always
    # traverse the full class hierarchy
    if hasattr(self, 'node_init'):
      for C in self.__class__.mro():
        if 'node_init' in C.__dict__:
          C.node_init(self)

  def __hash__(self):
    # print "Warning: __hash__ not implemented for %s" % self
    hash_value = 0x123123
    for m in self.members:
      v = getattr(self, m)
      if isinstance(v, np.ndarray):
        continue
      elif isinstance(v, (list, tuple)):
        for i in v: hash_value ^= hash(i)
      else:
        hash_value ^= hash(v)
    return hash_value

  def eq_members(self, other):
    print 'EQ:', self.__class__.__name__
    for (k,v) in self.iteritems():
      if not hasattr(other, k):
        return False
      if getattr(other, k) != v:
        return False
    return True

  def __eq__(self, other):
    return other.__class__ is  self.__class__ and self.eq_members(other)

  def __ne__(self, other):
    return not self == other

  def node_type(self):
    return self.__class__.__name__

  def clone(self, **kwds):
    cloned = copy.deepcopy(self)
    for (k,v) in kwds.values():
      setattr(cloned, k, v)
    return cloned

  def __str__(self):
    member_strings = []
    for (k,v) in self.iteritems():
      if isinstance(v, list):
        v_str = ['[']
        for i, v in enumerate(v):
          v_str.append('[%d] = %s' % (i, v))
        v_str += [']']

        v_str = '\n'.join(v_str)
      else:
        v_str = str(v)
      member_strings.append("%s = %s" % (k, v_str))
    child_str = '  ' + ',\n'.join(member_strings)
    child_str = child_str.replace('\n', '\n  ')

    return "%s { \n%s \n}" % (self.node_type(), child_str)

  def __repr__(self):
    return self.__str__()

  def iteritems(self):
    for k in self.members:
      yield (k, getattr(self, k, None))

  def itervalues(self):
    for (_,v) in self.iteritems():
      yield v

  def items(self):
    return [(k,getattr(self,k)) for k in self.members]


class Node(object):
  @classmethod
  def members(cls):
    return cls.members

  def __new__(cls, name, bases, attrs):
    members = set()
    for m in attrs.get('_members', []):
      members.add(m)

    for b in bases:
      for m in getattr(b, '_members', []):
        members.add(m)
    members = list(members)

    attrs['members'] = members
    attrs['__doc__'] = '\n'.join([':attribute %s: (initialized by Node)' % n for n in members])

    bases = list(bases)
    if object in bases:
      bases = [b for b in bases if b is not object]

    bases += [NodeTemplate]
    bases = tuple(bases)
    return type(name, bases, attrs)


