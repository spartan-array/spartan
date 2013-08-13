AM_CXX = clang++
AM_LD = clang++
AM_CXXFLAGS = -O2 -ggdb2 -Wall -Wno-unused-result -Wno-sign-compare -std=c++11 -fno-omit-frame-pointer

AM_LDFLAGS = -lboost_thread -lpthread
ACLOCAL_AMFLAGS = -I m4

SPARROW = sparrow

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
  sparrow/sparrow_service.h \
  pytable/_sparrow.cc
  
$(builddir)/pytable/_sparrow.cc : \
  $(srcdir)/pytable/sparrow.i $(SWIG_INCLUDES)
	swig -threads $(CPPFLAGS) -fvirtual -O -python -c++ -o $@ $< 
 
$(builddir)/sparrow/sparrow_service.h : $(srcdir)/sparrow/sparrow_service.rpc
	$(srcdir)/external/simple-rpc/rpc/rpcgen.py $<


lib_LTLIBRARIES = libsparrow.la librpc.la _sparrow.la
noinst_PROGRAMS = test-accum test-iterator

LDADD = libsparrow.la librpc.la
_sparrow_la_LDFLAGS = -module -lrt
_sparrow_la_LIBADD = libsparrow.la librpc.la

librpc_la_SOURCES = \
    external/simple-rpc/rpc/client.cc \
    external/simple-rpc/rpc/marshal.cc \
    external/simple-rpc/rpc/polling.cc \
    external/simple-rpc/rpc/server.cc \
    external/simple-rpc/rpc/utils.cc
	
libsparrow_la_SOURCES =\
	sparrow/util/stringpiece.cc\
	sparrow/util/registry.cc\
	sparrow/util/common.cc\
	sparrow/table.cc\
	sparrow/kernel.cc\
	sparrow/worker.cc\
	sparrow/master.cc


_sparrow_la_SOURCES =\
	pytable/_sparrow.cc \
    pytable/support.cc

test_accum_SOURCES = tests/test-accum.cc
test_iterator_SOURCES = tests/test-iterator.cc
