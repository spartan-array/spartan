SRCDIR := src/
OBJDIR := build/temp.linux-x86_64-2.7/

CPPFLAGS := -I${SRCDIR} -I${SRCDIR}/simple-rpc -I/usr/include/python2.7
CXXFLAGS := ${CPPFLAGS} -pthread -fno-strict-aliasing -DNDEBUG -g -fwrapv -O0 -Wall -fPIC -ggdb2 -std=c++0x
CXX ?= g++ 

SOURCES := $(shell find ${SRCDIR} -name '*.cc') spartan/wrap/spartan_wrap.cc
OBJS := $(addprefix ${OBJDIR},$(patsubst %.cc,%.o,$(SOURCES)))

_spartan_wrap.so: ${OBJS}
	${CXX} \
	-pthread -shared -Wl,-O1 -Wl,-Bsymbolic-functions \
	-Wl,-Bsymbolic-functions -Wl,-z,relro -fno-strict-aliasing \
	-DNDEBUG -g -fwrapv -O2 -Wall -Wstrict-prototypes \
	-D_FORTIFY_SOURCE=2 -g -fstack-protector --param=ssp-buffer-size=4 \
	-Wformat -Werror=format-security -o $@ $^

spartan/wrap/spartan_wrap.cc : spartan/wrap/spartan.i
	swig -python -Isrc -modern -O -c++ -threads -o $@ $^

setup:
	python setup.py develop --user

doc:
	python setup.py build_sphinx

clean:
	rm -rf build
	rm -rf build-opt

.PHONY: all doc clean

define OBJ_template

$(OBJDIR)$(patsubst %.cc,%.o,$(1)): $(1)
	mkdir -p $$(dir $$@)
	$(CXX) $(CXXFLAGS) -c $$< -o $$@

endef

$(foreach f,$(SOURCES), $(eval $(call OBJ_template,$f)))

