#!/usr/bin/env python
import argparse
import time

parser = argparse.ArgumentParser()
flags = argparse.Namespace()
_names = set() 

def parse_known_args(argv):
  parsed_flags, rest = parser.parse_known_args(argv)
  for flagname in _names:
    setattr(flags, flagname, getattr(parsed_flags, flagname))
    
  return flags, rest

def add_flag(name, *args, **kw):
  _names.add(name)
  parser.add_argument('--' + name, *args, **kw)
  
def add_bool_flag(name, default):
  _names.add(name)
  
  parser.add_argument('--' + name, default=default, type=bool, dest=name)
  parser.add_argument('--enable_' + name, action='store_true', dest=name)
  parser.add_argument('--disable_' + name, action='store_false', dest=name)

add_bool_flag('opt_fold', default=True)
add_bool_flag('opt_numexpr', default=True)
add_bool_flag('optimization', default=True)
add_bool_flag('profile_kernels', default=False)
add_flag('log_level', default=3, type=int)

HOSTS = [ ('localhost', 8) ]

#HOSTS = [
#  ('localhost', 4),
#  ('beaker-14', 8),
#  ('beaker-15', 8),
# # ('beaker-16', 8),
#  ('beaker-17', 8),
#  ('beaker-18', 8),
#  ('beaker-19', 8),
#  ('beaker-20', 8),
#  ('beaker-21', 8),
#  ('beaker-22', 8),
#  ('beaker-23', 8),
#]