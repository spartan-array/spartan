'''Helper for constructing trees of objects.

Provides pretty printing, equality testing, hashing and keyword initialization.
'''  


from itertools import izip
import copy

_members_cache = {}  
_mro_cache = {}
_reversed_mro_cache = {}
  
def get_mro(klass):
  if klass in _mro_cache:
    return _mro_cache[klass]
  else:
    mro = klass.mro()
    rev_mro = list(reversed(mro))
    _mro_cache[klass] = mro
    _reversed_mro_cache[klass] = rev_mro
    return mro 
  
def get_reverse_mro(klass):
  if klass in _reversed_mro_cache:
    return _reversed_mro_cache[klass]
  else:
    mro = klass.mro()
    rev_mro = list(reversed(mro))
    _mro_cache[klass] = mro
    _reversed_mro_cache[klass] = rev_mro
    return rev_mro 
  

def node_initializer(self, *args, **kw):
  get_members = self.members()
  n_args = len(args)
  n_members = len(get_members)
  class_name = self.__class__.__name__ 
  self_dict = self.__dict__
  if n_args == n_members:
    assert len(kw) == 0
    for (k,v) in zip(get_members,args):
      self_dict[k] = v
  elif n_args < n_members:
    for field in get_members:
      self_dict[field] = kw.get(field)
    
    for field, value in izip(get_members, args):
      self_dict[field] = value
      
    for (k,v) in kw.iteritems():
      assert k in get_members, \
        "Keyword argument '%s' not recognized for %s: %s" % \
        (k, self.node_type(), get_members)
  else:
    raise Exception('Too many arguments for %s, expected %s' % \
                    (class_name, get_members))
     
  # it's more common to not define a node initializer, 
  # so add an extra check to avoid having to always
  # traverse the full class hierarchy 
  if hasattr(self, 'node_init'):
    for C in _reversed_mro_cache[self.__class__]:
      if 'node_init' in C.__dict__:
        C.node_init(self)

    
def get_members(klass):
  """
  Walk through classes in mro order, accumulating member names.
  """
  if klass  in _members_cache:
    return _members_cache[klass]
  
  m = []
  for c in get_mro(klass):
    curr_members = getattr(c, '_members', []) 
    for name in curr_members:
      if name not in m:
        m.append(name)  
  _members_cache[klass] = m
  return m


class Node(object):
  @classmethod
  def members(cls):
    return get_members(cls)
  
  def __init__(self, *args, **kw):
    assert len(args) == 0, 'Node objects must be initialized with keywords.'
    node_initializer(self, *args, **kw)
 
  def iteritems(self):
    for k in self.members():
      yield (k, getattr(self, k, None))
      
  def itervalues(self):
    for (_,v) in self.iteritems():
      yield v 
  
  def items(self):
    return [(k,getattr(self,k)) for k in self.members()]

  def __hash__(self):
    # print "Warning: __hash__ not implemented for %s" % self
    hash_values = []
    for m in self.members():
      v = getattr(self, m)
      if isinstance(v, (list, tuple)):
        v = tuple(v)
      hash_values.append(v)
    return hash(tuple(hash_values))
  
  def eq_members(self, other):
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


def node_type(klass):
  '''Decorator to add node behavior to a class.'''
  def obj_init(self, *args, **kw):
    node_initializer(self, *args, **kw)
  
  mem = get_members(klass)
  obj_init.__name__ = '__init__'
  doc = 'Tree-like object.  Members:\n\n' + '\n'.join(':param %s: ' % k for k in mem)
  obj_init.__doc__ = doc
  
  return type(klass.__name__, (klass, Node), 
             { '__doc__' : doc,
              '__init__' : obj_init }) 
