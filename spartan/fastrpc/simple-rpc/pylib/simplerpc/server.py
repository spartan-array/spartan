import traceback

from simplerpc import _pyrpc
from simplerpc.marshal import Marshal

class MarshalWrap(object):
    def __init__(self, f, input_types, output_types):
        # f input: input_marshal (id only), f output: output_marshal (id only)
        self.f = f
        self.input_types = input_types
        self.output_types = output_types

    # def __del__(self):
    #     print "properly cleaned up!"

    def __call__(self, input_marshal_id):
        input_m = Marshal(id=input_marshal_id, should_release=False)

        input_values = []
        for input_ty in self.input_types:
            input_values += input_m.read_obj(input_ty),
        try:
            output = self.f(*input_values)
        except:
            traceback.print_exc()
            raise

        if len(self.output_types) == 0:
            # void rpc
            return 0 # mark as a NULL reply

        output_m = Marshal(should_release=False) # C++ code will release the marshal object
        if len(self.output_types) == 1:
            # single return value
            output_m.write_obj(output, self.output_types[0])
        else:
            # multiple return values
            for i in range(len(self.output_types)):
                output_m.write_obj(output[i], self.output_types[i])

        return output_m.id


class Server(object):
    def __init__(self, n_threads=1):
        self.id = _pyrpc.init_server(n_threads)
        self.func_ids = {} # rpc_id => func_ptr

    def __del__(self):
        all_rpc_ids = self.func_ids.keys()
        for rpc_id in all_rpc_ids:
            self.unreg(rpc_id)
        _pyrpc.fini_server(self.id)

    def __reg_func__(self, rpc_id, func, input_types, output_types):
        rpc_func = MarshalWrap(func, input_types, output_types)
        ret = _pyrpc.server_reg(self.id, rpc_id, rpc_func)
        if ret != 0:
            _pyrpc.helper_decr_ref(rpc_func)
        else:
            self.func_ids[rpc_id] = rpc_func
        return ret

    def enable_udp(self):
        _pyrpc.server_enable_udp(self.id)

    def reg_svc(self, svc):
        svc.__reg_to__(self)

    def unreg(self, rpc_id):
        _pyrpc.server_unreg(self.id, rpc_id)
        rpc_func = self.func_ids[rpc_id]
        del self.func_ids[rpc_id]
        _pyrpc.helper_decr_ref(rpc_func)

    def start(self, addr):
        return _pyrpc.server_start(self.id, addr)
