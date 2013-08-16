AM_CXX = clang++
AM_LD = clang++
CXXFLAGS = -std=c++11\
 -O3\
 -Wall\
 -Wno-unused-result\
 -Wno-sign-compare\
 -fno-omit-frame-pointer\
 -ffunction-sections\
 -fdata-sections

AM_LDFLAGS = -lboost_thread -lpthread -Wl,--gc-sections
ACLOCAL_AMFLAGS = -I m4

CPPFLAGS += \
  -I$(builddir)/ \
  -I$(srcdir)/ \
  -I$(srcdir)/external/simple-rpc \
  -I/usr/include/python2.7

SWIG_INCLUDES = \
  $(SPARROW)/table.h\
  $(SPARROW)/master.h\
  $(SPARROW)/worker.h\
  pytable/support.h

BUILT_SOURCES = \
  spartan/spartan_service.h \
  pytable/_spartan.cc
  
$(builddir)/pytable/_spartan.cc : \
  $(srcdir)/pytable/spartan.i $(SWIG_INCLUDES)
	swig -threads $(CPPFLAGS) -fvirtual -O -python -c++ -o $@ $< 
 
$(builddir)/spartan/spartan_service.h : $(srcdir)/spartan/spartan_service.rpc
	$(srcdir)/external/simple-rpc/rpc/rpcgen.py $<


lib_LTLIBRARIES = libspartan.la librpc.la _spartan.la
noinst_PROGRAMS = test-accum test-iterator

LDADD = libspartan.la librpc.la
_spartan_la_LDFLAGS = -module -lrt
_spartan_la_LIBADD = libspartan.la librpc.la

librpc_la_SOURCES = \
    external/simple-rpc/rpc/client.cc \
    external/simple-rpc/rpc/marshal.cc \
    external/simple-rpc/rpc/polling.cc \
    external/simple-rpc/rpc/server.cc \
    external/simple-rpc/rpc/utils.cc
	
libspartan_la_SOURCES =\
	spartan/util/stringpiece.cc\
	spartan/util/registry.cc\
	spartan/util/common.cc\
	spartan/table.cc\
	spartan/kernel.cc\
	spartan/worker.cc\
	spartan/master.cc


_spartan_la_SOURCES =\
	pytable/_spartan.cc \
    pytable/support.cc

test_accum_SOURCES = tests/test-accum.cc
test_iterator_SOURCES = tests/test-iterator.cc
