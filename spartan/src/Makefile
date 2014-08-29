CC = c++

CUR_DIR = $(shell pwd)
CORE_DIR = $(CUR_DIR)/core
ARRAY_DIR = $(CUR_DIR)/array
RPCDIR = $(CUR_DIR)/rpc
RPCBASE = $(RPCDIR)/base-utils
RPC = $(RPCDIR)/simple-rpc

PY_CFLAGS = $(shell python-config --cflags)
PY_LDFLAGS = $(shell python-config --ldflags)
ifndef $(NUMPY_FLAGS)
    NUMPY = $(shell python -c "import numpy; print '-I' + numpy.__path__[0] + '/core/include/numpy'")
endif

CFLAGS = -I$(CUR_DIR) -I$(CORE_DIR) -I$(ARRAY_DIR) -I$(RPC) -I$(RPCBASE) $(PY_CFLAGS) $(NUMPY) -pthread -std=c++0x -O2 # -g
LDFLAGS = -L$(RPCBASE)/build -L$(RPC)/build -L$(CURDIR) $(PY_LDFLAGS) -lbase -lpython2.7 -std=c++11 # -g
OBJS_WORKER = $(CUR_DIR)/core/worker.o
OBJS_ARRAY = $(ARRAY_DIR)/cslice.o $(ARRAY_DIR)/cextent.o $(ARRAY_DIR)/ctile.o $(ARRAY_DIR)/carray.o $(ARRAY_DIR)/carray_reducer.o
OBJS_CORE = $(CUR_DIR)/core/cconfig.o
APPS = rpc libcore.so libspartan_array.so worker

all: $(APPS)

.cc.o:
	$(CC) $(CFLAGS) $< -c -o $@ -fPIC

.PHONY:base-utils simple-rpc
base-utils:
	@if [ ! -d "$(RPCBASE)" ] ; then \
	    git clone https://github.com/santazhang/base-utils $(RPCBASE); \
	    cd $(RPCBASE) && patch -Np1 < ../base-utils-patch 2> /dev/null && cd - ;\
	fi

simple-rpc:
	@if [ ! -d "$(RPC)" ] ; then \
	    git clone https://github.com/santazhang/simple-rpc $(RPC); \
	fi

rpc: base-utils simple-rpc
	cd $(RPCBASE);python waf configure;python waf;
	cd $(RPC);python waf configure;python waf;
	#cd $(RPCDIR);python simple-rpc/bin/rpcgen service.rpc --python --cpp;

libcore.so: $(OBJS_CORE)
	${CC} -o $(CUR_DIR)/$@ $^ $(LDFLAGS) -shared -fPIC -Wl,-init,_attach

libspartan_array.so: $(OBJS_ARRAY)
	${CC} -o $(CUR_DIR)/$@ $^ $(LDFLAGS) -shared -fPIC -Wl,-init,_attach -lcore

worker: $(OBJS_WORKER)
	${CC} -o $(CUR_DIR)/$@ $^ $(LDFLAGS) -lsimplerpc -lspartan_array -lcore

clean:
	cd $(RPCBASE);python waf clean
	cd $(RPC);python waf clean
	rm -f $(RPCDIR)/*.o
	rm -f $(ARRAY_DIR)/*.o
	rm -f $(CORE_DIR)/*.o
	rm -f $(CUR_DIR)/*.o
	rm -f $(CUR_DIR)/*.so $(CUR_DIR)/worker

