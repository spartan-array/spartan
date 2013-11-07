#!/usr/bin/env python

"""
Configuration options and flags.

Options may be specified on the command line, or via a configuration
file.  Configuration files should be placed in $HOME/.config/spartanrc.

"""

import argparse
import logging
import time
import os
import sys
import traceback
from spartan import util

HOSTS = [
  ('localhost', 8),
  ]


parser = argparse.ArgumentParser()
_names = set()

def add_flag(name, *args, **kw):
  _names.add(kw.get('dest', name))
  
  parser.add_argument('--' + name, *args, **kw)
  if 'default' in kw:
    return kw['default']
  return None
  
def add_bool_flag(name, default, **kw):
  _names.add(name)
  
  parser.add_argument('--' + name, default=default, type=int, dest=name, **kw)
  parser.add_argument('--enable_' + name, action='store_true', dest=name)
  parser.add_argument('--disable_' + name, action='store_false', dest=name)
  
  return default
  

class AssignMode(object):
  BY_CORE = 1
  BY_NODE = 2

class Flags(object):
  _parsed = False

  profile_kernels = add_bool_flag('profile_kernels', default=False)
  profile_master = add_bool_flag('profile_master', default=False)
  log_level = add_flag('log_level', default='INFO', type=str)
  num_workers = add_flag('num_workers', default=3, type=int)
  cluster = add_bool_flag('cluster', default=False)
  oprofile = add_bool_flag('oprofile', default=False)
  
  config_file = add_flag('config_file', default='', type=str)
  
  port_base = add_flag('port_base', type=int, default=10000,
    help='Port to listen on (master = port_base, workers=port_base + N)')
  
  use_threads = add_bool_flag(
    'use_threads',
    help='When running locally, use threads instead of forking. (slow, for debugging)', 
    default=False)
  
  assign_mode = AssignMode.BY_NODE
  add_flag('bycore', dest='assign_mode', action='store_const', const=AssignMode.BY_CORE)
  add_flag('bynode', dest='assign_mode', action='store_const', const=AssignMode.BY_NODE)
  
  def __getattr__(self, name):
    assert self.__dict__['_parsed'], 'Flags accessed before parse_args called.'
    return self.__dict__[name]
  
  def __repr__(self):
    result = []
    for k, v in iter(self):
      result.append('%s : %s' % (k, v))
    return '\n'.join(result)
  
  def __iter__(self):
    return iter([(k, getattr(self, k)) for k in dir(self)
                 if not k.startswith('_')])

flags = Flags()

def parse_args(argv):
  # force configuration settings to load.
  import spartan.expr
  import spartan.expr.optimize

  parsed_flags, rest = parser.parse_known_args(argv)
  for flagname in _names:
    setattr(flags, flagname, getattr(parsed_flags, flagname))
 
  flags._parsed = True

  logging.basicConfig(format='%(filename)s:%(lineno)s [%(funcName)s] %(message)s',
                      level=getattr(logging, flags.log_level))


  if flags.config_file == '':
    try:
      import appdirs
      flags.config_file = appdirs.user_data_dir('Spartan', 'rjpower.org') + '/spartanrc'

      if not os.path.exists(flags.config_file):
        os.makedirs(os.path.dirname(flags.config_file), mode=0755)
        open(flags.config_file, 'a').close()
    except:
      print 'Missing appdirs package; spartanrc will not be processed.'

  if flags.config_file:
    util.log_info('Loading configuration from %s' % (flags.config_file))
    try:
      eval(compile(open(flags.config_file).read(),
                   flags.config_file,
                   'exec'))
    except:
      util.log_fatal('Failed to parse config file: %s' % (flags.config_file),
                     exc_info=1)
      sys.exit(1)

  util.log_info('Hostlist: %s', HOSTS)

  return rest

