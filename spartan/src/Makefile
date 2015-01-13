CC = c++

CUR_DIR := $(shell pwd)
CORE_DIR := $(CUR_DIR)/core
CORE_DIR := $(CUR_DIR)/core
ARRAY_DIR := $(CUR_DIR)/array
RPCDIR := $(CUR_DIR)/rpc
RPC := $(RPCDIR)/simple-rpc
OBJ_DIR := $(CUR_DIR)/obj

PY_CFLAGS = $(shell python-config --cflags)
PY_LDFLAGS = $(shell python-config --ldflags)
ifndef $(NUMPY_FLAGS)
    NUMPY = $(shell python -c "import numpy; print '-I' + numpy.__path__[0] + '/core/include/numpy'")
endif

CFLAGS := -I$(CUR_DIR) -I$(CORE_DIR) -I$(ARRAY_DIR) -I$(RPC) $(PY_CFLAGS) $(NUMPY) -pthread -std=c++0x -O2
LDFLAGS := -L$(RPC)/build -L$(OBJ_DIR) -L$(CURDIR) -L$(RPC) $(PY_LDFLAGS) -lbase -lpython2.7 -std=c++11

OBJS_WORKER = $(CUR_DIR)/core/worker.o
OBJS_ARRAY = $(ARRAY_DIR)/cslice.o $(ARRAY_DIR)/cextent.o $(ARRAY_DIR)/ctile.o $(ARRAY_DIR)/carray.o $(ARRAY_DIR)/carray_reducer.o
OBJS_CORE = $(CUR_DIR)/core/cconfig.o

all: rpc $(OBJ_DIR)/libcore.so $(OBJ_DIR)/libspartan_array.so $(OBJ_DIR)/worker

.cc.o:
	$(CC) $(CFLAGS) $< -c -o $@ -fPIC
	$(CC) -MM $(CFLAGS) -MT '$@' $< > $(<:.cc=.d)

.PHONY:simple-rpc
simple-rpc:
	@if [ ! -d "$(RPC)" ] ; then \
	    git clone https://github.com/santazhang/simple-rpc $(RPC); \
	    cp $(RPCDIR)/simple-rpc-wscript $(RPC)/wscript; \
			cd $(RPCDIR) && patch -Np1 < $(CUR_DIR)/rpc/base-patch 2> /dev/null && cd -; \
	fi

rpc: simple-rpc
	cd $(RPC);python waf configure; python waf;
	@if [ ! -f "$(RPCDIR)/service.h" ] ; then \
			cd $(RPCDIR);python simple-rpc/bin/rpcgen service.rpc --python --cpp; \
	fi
	@mkdir -p $(OBJ_DIR)
	mv $(RPC)/build/*.so $(OBJ_DIR)

-include $(OBJS_CORE:.o=.d)
$(OBJ_DIR)/libcore.so: $(OBJS_CORE)
	${CC} -o $@ $^ $(LDFLAGS) -shared -fPIC -Wl,-init,_attach

-include $(OBJS_ARRAY:.o=.d)
$(OBJ_DIR)/libspartan_array.so: $(OBJS_ARRAY)
	${CC} -o $@ $^ $(LDFLAGS) -shared -fPIC -Wl,-init,_attach -lcore

-include $(OBJS_WORKER:.o=.d)
$(OBJ_DIR)/worker: $(OBJS_WORKER)
	${CC} -o $@ $^ $(LDFLAGS) -lsimplerpc -lspartan_array -lcore

#TODO: This needs to be updated
clean:
	rm -f $(RPCDIR)/*.o $(RPCDIR)/*.d
	rm -f $(ARRAY_DIR)/*.o $(ARRAY_DIR)/*.d
	rm -f $(CORE_DIR)/*.o $(CORE_DIR)/*.d
	rm -rf $(RPCDIR)/simple-rpc $(OBJ_DIR)
	rm -f $(RPCDIR)/service.h $(RPCDIR)/service.py
