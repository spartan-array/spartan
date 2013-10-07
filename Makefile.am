CXXFLAGS = -std=c++11\
 -O2\
 -ggdb2\
 -Wall\
 -Wno-unused-result\
 -Wno-sign-compare\
 -fno-omit-frame-pointer

AM_LDFLAGS = -lpthread
ACLOCAL_AMFLAGS = -I m4

CPPFLAGS += \
  -I$(builddir)/ \
  -I$(srcdir)/ \
  -I$(srcdir)/python \
  -I$(srcdir)/external/simple-rpc \
  -I$(srcdir)/external/\
  -I/usr/include/python2.7

SWIG_INCLUDES = \
  spartan/table.h\
  spartan/master.h\
  spartan/worker.h

BUILT_SOURCES = \
  spartan/spartan_service.h \
  $(builddir)/python/spartan/_spartan_wrap.cc
  
$(builddir)/python/spartan/_spartan_wrap.cc : \
  $(srcdir)/python/spartan/spartan.i $(SWIG_INCLUDES)
	swig -threads $(CPPFLAGS) -fvirtual -O -python -c++ -o $@ $< 
 
$(builddir)/spartan/spartan_service.h : $(srcdir)/spartan/spartan_service.rpc
	$(srcdir)/external/simple-rpc/rpc/rpcgen.py $<


noinst_LTLIBRARIES = libspartan.la librpc.la 
lib_LTLIBRARIES = _spartan_wrap.la

LDADD = libspartan.la librpc.la
_spartan_wrap_la_LDFLAGS = -module -lrt
_spartan_wrap_la_LIBADD = libspartan.la librpc.la

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


_spartan_wrap_la_SOURCES =\
	python/spartan/_spartan_wrap.cc \
    python/spartan/support.cc

