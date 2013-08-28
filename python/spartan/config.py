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

def add_flag(name, default, *args, **kw):
  _names.add(name)
  parser.add_argument('--' + name, default=default, *args, **kw)
  
def add_bool_flag(name, default):
  _names.add(name)
  parser.add_argument('--' + name, default=default, action='store', dest=name, type=bool)
  parser.add_argument('--enable_' + name, default=default, action='store_true', dest=name)
  parser.add_argument('--disable_' + name, default=default, action='store_false', dest=name)

add_bool_flag('folding', default=True)
add_bool_flag('optimization', default=True)
add_bool_flag('profile_kernels', default=False)