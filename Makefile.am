CXX = mpic++
LD = mpic++
CXXFLAGS = -Wall -std=c++11 -O2 -fno-omit-frame-pointer

AM_LDFLAGS =  -Wl,--export-dynamic

LDFLAGS = -no-fast-install \
  -L/home/power/pkg/openmpi-1.7/lib -lmpi_cxx -lmpi -lopen-rte -lopen-pal -lm -ldl -lrdmacm -libverbs -lrt -lnsl -lutil -lprotobuf -lboost_thread


ACLOCAL_AMFLAGS = -I m4

CPPFLAGS += \
  -I$(builddir)/src/ \
  -I$(srcdir)/src/ \
  -I$(srcdir)/src/external/google-flags \
  -I$(srcdir)/src/external/google-logging \
  -I/home/power/pkg/openmpi-1.7/include \
  -I/usr/include/python2.7


BUILT_SOURCES = \
  src/sparrow/sparrow.pb.h \
  src/sparrow/python/_sparrow.cc
  
$(builddir)/src/sparrow/python/_sparrow.cc : $(srcdir)/src/sparrow/python/sparrow.i
	swig -O -python -c++ -o $@ $< 
 
$(builddir)/src/sparrow/sparrow.pb.cc $(builddir)/src/sparrow/sparrow.pb.h : $(srcdir)/src/sparrow/sparrow.proto
	protoc --proto_path=$(srcdir)/src/sparrow --cpp_out=$(dir $@) $^


noinst_LIBRARIES = _sparrow.so
lib_LTLIBRARIES = libsparrow.la libgflags.la libglog.la libpysparrow.la
noinst_PROGRAMS = test-put test-accum

LDADD = libsparrow.la libgflags.la libglog.la -lprotobuf -lboost_thread -lpthread

libgflags_la_SOURCES = \
	src/external/google-flags/gflags.cc\
	src/external/google-flags/gflags_reporting.cc\
	src/external/google-flags/gflags_nc.cc
	
libglog_la_SOURCES = \
	src/external/google-logging/utilities.cc\
	src/external/google-logging/vlog_is_on.cc\
	src/external/google-logging/demangle.cc\
	src/external/google-logging/logging.cc\
	src/external/google-logging/raw_logging.cc\
	src/external/google-logging/symbolize.cc\
	src/external/google-logging/signalhandler.cc
	
libsparrow_la_SOURCES =\
    src/sparrow/sparrow.pb.cc\
	src/sparrow/util/stringpiece.cc\
	src/sparrow/util/file.cc\
	src/sparrow/util/registry.cc\
	src/sparrow/util/rpc.cc\
	src/sparrow/util/common.cc\
	src/sparrow/table.cc\
	src/sparrow/kernel.cc\
	src/sparrow/worker.cc\
	src/sparrow/master.cc

libpysparrow_la_LIBADD = libsparrow.la libgflags.la libglog.la -lprotobuf -lboost_thread -lpthread
libpysparrow_la_SOURCES =\
	src/sparrow/python/_sparrow.cc \
    src/sparrow/python/support.cc

_sparrow_so_SOURCES =


_sparrow.so: libpysparrow.la
	cp .libs/libpysparrow.so _sparrow.so


test_put_SOURCES = src/tests/test-put.cc
test_accum_SOURCES = src/tests/test-accum.cc
