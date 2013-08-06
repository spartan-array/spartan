AM_CXX = mpic++
AM_LD = mpic++
AM_CXXFLAGS = -Wall -Wno-sign-compare -std=c++11 -O2 -fno-omit-frame-pointer -ggdb2

MPI_LDFLAGS = -pthread -L/home/power/pkg/openmpi-1.7/lib -lmpi_cxx -lmpi -lopen-rte -lopen-pal -lm -ldl -lrdmacm -libverbs -lrt -lnsl -lutil -Wl,--export-dynamic
AM_LDFLAGS = $(MPI_LDFLAGS) -lprotobuf -lboost_thread -lpthread
ACLOCAL_AMFLAGS = -I m4

SPARROW = sparrow

CPPFLAGS += \
  -I$(builddir)/ \
  -I$(srcdir)/ \
  -I$(srcdir)/external/google-flags \
  -I$(srcdir)/external/google-logging \
  -I/home/power/pkg/openmpi-1.7/include \
  -I/usr/include/python2.7

SWIG_INCLUDES = \
  $(SPARROW)/table.h\
  $(SPARROW)/master.h\
  $(SPARROW)/worker.h\
  pytable/support.h

BUILT_SOURCES = \
  sparrow/sparrow.pb.h \
  pytable/_sparrow.cc
  
$(builddir)/pytable/_sparrow.cc : \
  $(srcdir)/pytable/sparrow.i $(SWIG_INCLUDES)
	swig $(CPPFLAGS) -fvirtual -O -python -c++ -o $@ $< 
 
$(builddir)/sparrow/sparrow.pb.cc $(builddir)/sparrow/sparrow.pb.h : $(srcdir)/sparrow/sparrow.proto
	protoc --proto_path=$(srcdir)/sparrow --cpp_out=$(dir $@) $^


lib_LTLIBRARIES = libsparrow.la libgflags.la libglog.la _sparrow.la
noinst_PROGRAMS = test-accum test-iterator

LDADD = libsparrow.la libgflags.la libglog.la
_sparrow_la_LDFLAGS = -module
_sparrow_la_LIBADD = libsparrow.la libgflags.la libglog.la

libgflags_la_SOURCES = \
	external/google-flags/gflags.cc\
	external/google-flags/gflags_reporting.cc\
	external/google-flags/gflags_nc.cc
	
libglog_la_SOURCES = \
	external/google-logging/utilities.cc\
	external/google-logging/vlog_is_on.cc\
	external/google-logging/demangle.cc\
	external/google-logging/logging.cc\
	external/google-logging/raw_logging.cc\
	external/google-logging/symbolize.cc\
	external/google-logging/signalhandler.cc
	
libsparrow_la_SOURCES =\
    sparrow/sparrow.pb.cc\
	sparrow/util/stringpiece.cc\
	sparrow/util/file.cc\
	sparrow/util/registry.cc\
	sparrow/util/rpc.cc\
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
