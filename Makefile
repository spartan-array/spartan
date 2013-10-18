dbg:
	mkdir -p build-dbg
	test -f build-dbg/Makefile || (cd build-dbg && ../configure CFLAGS=-O0 CXXFLAGS=-O0)
	cd build-dbg && $(MAKE)
	rm build
	ln -sf build-dbg build

opt:
	mkdir -p build-opt
	test -f build-opt/Makefile || (cd build-opt && ../configure)
	cd build-opt && $(MAKE)
	rm build
	ln -sf build-opt build

clean:
	rm -rf build-opt
	rm -rf build-dbg
