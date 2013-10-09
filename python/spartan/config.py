#!/usr/bin/env python
import argparse
import time

parser = argparse.ArgumentParser()
_names = set() 

def add_flag(name, *args, **kw):
  _names.add(name)
  parser.add_argument('--' + name, *args, **kw)
  if 'default' in kw:
    return kw['default']
  return None
  
def add_bool_flag(name, default):
  _names.add(name)
  
  parser.add_argument('--' + name, default=default, type=bool, dest=name)
  parser.add_argument('--enable_' + name, action='store_true', dest=name)
  parser.add_argument('--disable_' + name, action='store_false', dest=name)
  
  return default

class Flags(object):
  opt_fold = add_bool_flag('opt_fold', True)
  opt_numexpr = add_bool_flag('opt_numexpr', False)
  opt = add_bool_flag('optimization', default=True)
  profile_kernels = add_bool_flag('profile_kernels', default=False)
  profile_master = add_bool_flag('profile_master', default=False)
  log_level = add_flag('log_level', default=3, type=int)
  num_workers = add_flag('num_workers', default=4, type=int)
  cluster = add_bool_flag('cluster', default=False)

flags = Flags()

def parse_known_args(argv):
  parsed_flags, rest = parser.parse_known_args(argv)
  for flagname in _names:
    setattr(flags, flagname, getattr(parsed_flags, flagname))
    
  return flags, rest

#HOSTS = [ ('localhost', 8) ]

HOSTS = [
  ('beaker-14', 8),
  ('beaker-15', 8),
 # ('beaker-16', 8),
  ('beaker-17', 8),
  ('beaker-18', 8),
  ('beaker-19', 8),
  ('beaker-20', 8),
  ('beaker-21', 8),
  ('beaker-22', 8),
  ('beaker-23', 8),
  ('beaker-24', 8),
  ('beaker-25', 8),
]
