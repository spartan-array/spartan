#!/usr/bin/env python

"""
Configuration options and flags.

Options may be specified on the command line, or via a configuration
file.  Configuration files should be placed in $HOME/.config/spartan.ini

"""
import ConfigParser
import argparse
import logging
import os
import sys

import appdirs

from spartan import util

class Flag(object):
  def __init__(self, name, default=None, help=''):
    self.name = name
    self.val = default
    self.help = help

  def __repr__(self):
    return '--%s=%s' % (self.name, self._str())

  def _str(self):
    return str(self.val)


class IntFlag(Flag):
  def parse(self, str):
    self.val = int(str)

class StrFlag(Flag):
  def parse(self, str):
    self.val = str

class BoolFlag(Flag):
  def parse(self, str):
    str = str.lower()
    str = str.strip()

    if str == 'false' or str == '0': val = False
    else: val = True
    #print 'Bool %s "%s" %s' % (self.name, str, val)
    self.val = val

  def _str(self):
    return str(int(self.val))


LOG_STR = {logging.DEBUG: 'DEBUG',
           logging.INFO: 'INFO',
           logging.WARN: 'WARN',
           logging.ERROR: 'ERROR',
           logging.FATAL: 'FATAL'}


class LogLevelFlag(Flag):
  def parse(self, str):
    self.val = getattr(logging, str)

  def _str(self):
    return LOG_STR[self.val]



class Flags(object):
  def __init__(self):
    self._parsed = False
    self._vals = {}

  def add(self, flag):
    self._vals[flag.name] = flag

  def __getattr__(self, key):
    if key.startswith('_'): return self.__dict__[key]

    assert self.__dict__['_parsed'], 'Access to flags before config.initialize() called.'
    return self.__dict__['_vals'][key].val

  def __repr__(self):
    return ' '.join([repr(f) for f in self._vals.values()])

  def __str__(self):
    return repr(self)

  def __iter__(self):
    return iter(self._vals.items())


FLAGS = Flags()

FLAGS.add(BoolFlag('profile_kernels', default=False))
FLAGS.add(BoolFlag('profile_master', default=False))
FLAGS.add(BoolFlag('cluster', default=False))
FLAGS.add(LogLevelFlag('log_level', logging.INFO))
FLAGS.add(IntFlag('num_workers', default=3))
FLAGS.add(IntFlag('port_base', default=10000,
                  help='Port to listen on (master = port_base, workers=port_base + N)'))

def initialize(argv):
  '''Parse configuration from flags and/or configuration file.'''

  import spartan.expr.optimize
  import spartan.cluster

  # load flags from other packages (is there a better way to do this?)
  if FLAGS._parsed:
    return

  FLAGS._parsed = True

  config_file = appdirs.user_data_dir('Spartan', 'rjpower.org') + '/spartan.ini'
  config_dir = os.path.dirname(config_file)
  if not os.path.exists(config_dir):
    os.makedirs(config_dir, mode=0755)

  if not os.path.exists(config_file):
    open(config_file, 'a').close()

  print >>sys.stderr, 'Loading configuration from %s' % (config_file)
  try:
    config = ConfigParser.ConfigParser()
    config.read(config_file)

    if config.has_section('flags'):
      for name, value in config.items('flags'):
        argv.append('--%s=%s' % (name, value))
  except:
    print >>sys.stderr, 'Failed to parse config file: %s' % config_file
    sys.exit(1)

  parser = argparse.ArgumentParser()
  for name, flag in FLAGS:
    parser.add_argument('--' + name, type=str)

  parsed_flags, rest = parser.parse_known_args(argv)
  for name, flag in FLAGS:
    if getattr(parsed_flags, name) is not None:
      flag.parse(getattr(parsed_flags, name))

  logging.basicConfig(format='%(filename)s:%(lineno)s [%(funcName)s] %(message)s',
                      level=FLAGS.log_level,
                      stream=sys.stderr)

  for f in rest:
    if f.startswith('-'):
      util.log_warn('Unknown flag: %s (ignored)' % f)

  util.log_debug('Hostlist: %s', FLAGS.hosts)
  return rest

