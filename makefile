CC = c++

NEW_CFLAGS = $(shell python-config --cflags)
NEW_LDFLAGS = $(shell python-config --ldflags)

CUR_DIR = $(shell pwd)
MAINDIR = spartan/
RPCDIR = spartan/fastrpc/
NUMPY = $(shell python -c "import numpy; print '-I' + numpy.__path__[0] + '/core/include/numpy'")
RPC = $(RPCDIR)/simple-rpc
RPCBASE = $(RPCDIR)/base-utils
CFLAGS = $(NUMPY) -I$(RPC) -I$(RPCBASE) -L$(RPC)/build $(NEW_CFLAGS) -pthread -lsimplerpc -std=c++0x -g
LDFLAGS = -L$(RPCBASE)/build -L$(RPC)/build $(NEW_LDFLAGS) -lbase -lsimplerpc -lpython2.7 -std=c++11 -g

OBJS_WORKER = $(MAINDIR)/worker.o $(MAINDIR)/cconfig.o
OBJS_TEST = $(MAINDIR)/test.o
APPS = rpc worker

all: $(APPS)
		#export LD_LIBRARY_PATH=/home/chenqi/workspace/spartan/spartan/fastrpc/simple-rpc/build/:$(LD_LIBRARY_PATH)

.cc.o:
		$(CC) $(CFLAGS) $< -c -o $@

rpc:
		cd $(RPCBASE);python waf configure;python waf;
		cd $(RPC);python waf configure;python waf;
		cd $(RPCDIR);python simple-rpc/bin/rpcgen service.rpc --python --cpp;
		python setup.py develop --user;

worker: $(OBJS_WORKER)
		${CC} -o $(MAINDIR)/$@ $^ $(LDFLAGS) $(CFLAGS)

test: $(OBJS_TEST)
		${CC} -o $(MAINDIR)/$@ $^ $(LDFLAGS) $(CFLAGS)

clean:
		cd $(RPCBASE);python waf clean
		cd $(RPC);python waf clean
		rm -f $(RPCDIR)/*.o $(MAINDIR)/*.o $(MAINDIR)/*.cpp $(MAINDIR)/*.so
		for APP in "worker"; do \
			rm -f $(MAINDIR)/$$APP; \
		done

