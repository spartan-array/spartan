CXX = mpic++

CPPFLAGS = \
  -I$(srcdir)/src/ \
  -I$(srcdir)/src/external/google-flags \
  -I$(srcdir)/src/external/google-logging

sparrow.pb.cc sparrow.pb.h : src/sparrow.proto
	protoc --proto_path=$(dir $^) --cpp_out=$(dir $^) $^

#BUILT_SOURCES = sparrow.pb.h

bin_PROGRAMS = sparrow
sparrow_SOURCES =\
	src/util/stringpiece.cc\
	src/util/file.cc\
	src/util/registry.cc\
	src/util/rpc.cc\
	src/util/common.cc\
	src/kernel/global-table.cc\
	src/worker/worker.cc\
	src/master/master.cc\
	src/external/google-flags/gflags.cc\
	src/external/google-flags/gflags_reporting.cc\
	src/external/google-flags/gflags_nc.cc\
	src/external/google-logging/utilities.cc\
	src/external/google-logging/vlog_is_on.cc\
	src/external/google-logging/demangle.cc\
	src/external/google-logging/logging.cc\
	src/external/google-logging/symbolize.cc\
	src/external/google-logging/signalhandler.cc
