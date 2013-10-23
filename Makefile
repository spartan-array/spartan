ifdef _REALBUILD
SRCDIR := ../src

CPPFLAGS := -I${SRCDIR} -I${SRCDIR}/simple-rpc -I/usr/include/python2.7
CXXFLAGS := ${CPPFLAGS} -pthread -fno-strict-aliasing -DNDEBUG -g -fwrapv -O2 -Wall -fPIC -ggdb2 -std=c++0x
CXX ?= g++ 

spartan_wrap.cc : ../spartan/wrap/spartan.i
	swig -python -Isrc -modern -O -c++ -threads -o $@ $^

SOURCES := $(shell find ${SRCDIR} -name '*.cc') spartan_wrap.cc
OBJS := $(notdir $(patsubst %.cc,%.o,${SOURCES}))
VPATH := ${SRCDIR} ${SRCDIR}/spartan ${SRCDIR}/spartan/util ${SRCDIR}/simple-rpc/rpc

%.o : %.cc
	${CXX} ${CXXFLAGS} -c $^ -o $@

_spartan_wrap.so: ${OBJS}
	${CXX} \
	-pthread -shared -Wl,-O1 -Wl,-Bsymbolic-functions \
	-Wl,-Bsymbolic-functions -Wl,-z,relro -fno-strict-aliasing \
	-DNDEBUG -g -fwrapv -O2 -Wall -Wstrict-prototypes \
	-D_FORTIFY_SOURCE=2 -g -fstack-protector --param=ssp-buffer-size=4 \
	-Wformat -Werror=format-security -o $@ $^

else

all:
	#mkdir -p build-opt
	#cd build-opt && _REALBUILD=1 $(MAKE) -f../Makefile _spartan_wrap.so
	python setup.py develop --user

doc:
	python setup.py build_sphinx

clean:
	rm -rf build
	rm -rf build-opt

.PHONY: all doc clean

endif