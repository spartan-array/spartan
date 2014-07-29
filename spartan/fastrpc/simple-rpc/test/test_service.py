import os
from simplerpc.marshal import Marshal
from simplerpc.future import Future

empty_struct = Marshal.reg_type('empty_struct', [])

complex_struct = Marshal.reg_type('complex_struct', [('d', 'std::map<std::pair<std::string, std::string>, std::vector<std::vector<std::pair<std::string, std::string>>>>'), ('s', 'std::set<std::string>'), ('e', 'empty_struct')])

class EmptyService(object):

    __input_type_info__ = {
    }

    __output_type_info__ = {
    }

    def __bind_helper__(self, func):
        def f(*args):
            return getattr(self, func.__name__)(*args)
        return f

    def __reg_to__(self, server):
        pass

class EmptyProxy(object):
    def __init__(self, clnt):
        self.__clnt__ = clnt

class MathService(object):
    GCD = 0x475dd711

    __input_type_info__ = {
        'gcd': ['rpc::i64','rpc::i64'],
    }

    __output_type_info__ = {
        'gcd': ['rpc::i64'],
    }

    def __bind_helper__(self, func):
        def f(*args):
            return getattr(self, func.__name__)(*args)
        return f

    def __reg_to__(self, server):
        server.__reg_func__(MathService.GCD, self.__bind_helper__(self.gcd), ['rpc::i64','rpc::i64'], ['rpc::i64'])

    def gcd(__self__, a, in1):
        raise NotImplementedError('subclass MathService and implement your own gcd function')

class MathProxy(object):
    def __init__(self, clnt):
        self.__clnt__ = clnt

    def async_gcd(__self__, a, in1):
        return __self__.__clnt__.async_call(MathService.GCD, [a, in1], MathService.__input_type_info__['gcd'], MathService.__output_type_info__['gcd'])

    def sync_gcd(__self__, a, in1):
        __result__ = __self__.__clnt__.sync_call(MathService.GCD, [a, in1], MathService.__input_type_info__['gcd'], MathService.__output_type_info__['gcd'])
        if __result__[0] != 0:
            raise Exception("RPC returned non-zero error code %d: %s" % (__result__[0], os.strerror(__result__[0])))
        if len(__result__[1]) == 1:
            return __result__[1][0]
        elif len(__result__[1]) > 1:
            return __result__[1]

