#Simple script to invoke Makefiel in /src and move executable into appropriate places
CUR_DIR := $(shell pwd)
SRC_DIR := $(CUR_DIR)/src
OBJ_DIR := $(SRC_DIR)/obj
LIB_DIR := $(CUR_DIR)/lib
RPC_DIR := $(CUR_DIR)/rpc
SIM_DIR := $(RPC_DIR)/simplerpc
all: dir_mk mv_data

#Special var to keep track of executable from src/ to spartan/
PKG_DATA := worker

dir_mk:
	@mkdir -p $(LIB_DIR)
	@mkdir -p $(SIM_DIR)

mv_data:
	make -C $(SRC_DIR)
	@cp $(OBJ_DIR)/pkg/* $(CUR_DIR)
	@cp $(OBJ_DIR)/lib/* $(LIB_DIR)
	@cp $(OBJ_DIR)/rpc/* $(SIM_DIR)

clean:
	make clean -C $(SRC_DIR)
	rm -rf $(LIB_DIR)
	rm -rf $(SIM_DIR)
	rm -rf $(PKG_DATA)

cleanall:
	make cleanall -C $(SRC_DIR)
	rm -rf $(LIB_DIR)
	rm -rf $(SIM_DIR)
	rm -rf $(PKG_DATA)
