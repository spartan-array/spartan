CC = c++

MAINDIR = spartan/
RPCDIR = spartan/fastrpc/
NUMPY = $(shell python -c "import numpy; print '-I' + numpy.__path__[0] + '/core/include/numpy'")
RPC = $(RPCDIR)/simple-rpc
RPCBASE = $(RPCDIR)/base-utils
CFLAGS = $(NUMPY) -I$(RPC) -I$(RPCBASE) $(shell python-config --cflags) -pthread -std=c++0x
LDFLAGS = -L$(RPCBASE)/build -L$(RPC)/build $(shell python-config --ldflags) -lsimplerpc -lbase -std=c++11

OBJS_WORKER = $(MAINDIR)/worker.o
OBJS_MASTER = $(MAINDIR)/master.o
OBJS_CLIENT = $(MAINDIR)/client.o
OBJS_TEST = $(MAINDIR)/test.o
APPS = worker master client test

all: $(APPS)

.cc.o:
		$(CC) $(CFLAGS) $< -c -o $@

worker: $(OBJS_WORKER)
		${CC} -o $(MAINDIR)/$@ $^ $(LDFLAGS) $(CFLAGS)

master: $(OBJS_MASTER)
		${CC} -o $(MAINDIR)/$@ $^ $(LDFLAGS) $(CFLAGS)

client: $(OBJS_CLIENT)
		${CC} -o $(MAINDIR)/$@ $^ $(LDFLAGS) $(CFLAGS)

test: $(OBJS_TEST)
		${CC} -o $(MAINDIR)/$@ $^ $(LDFLAGS) $(CFLAGS)

clean:
		rm -f $(RPCDIR)/*.o $(MAINDIR)/*.o
		for APP in $(APPS); do \
			rm -f $(MAINDIR)/$$APP; \
		done
