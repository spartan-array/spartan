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
LDFLAGS := -L$(RPC)/build -L$(CURDIR) -L$(RPC) $(PY_LDFLAGS) -lpython2.7 -std=c++11

OBJ_DIRS :=

all:

.cc.o:
	$(CC) $(CFLAGS) $< -c -o $@ -fPIC
	$(CC) -MM $(CFLAGS) -MT '$@' $< > $(<:.cc=.d)

include rpc/Makefrag
