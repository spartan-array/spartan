import os
from simplerpc.marshal import Marshal
from simplerpc.future import Future

point3 = Marshal.reg_type('point3', [('x', 'double'), ('y', 'double'), ('z', 'double')])

class BenchmarkService(object):
    FAST_PRIME = 0x5b9365eb
    FAST_DOT_PROD = 0x4a92c615
    FAST_ADD = 0x484806bf
    FAST_NOP = 0x4f29b7a5
    PRIME = 0x1f96b90f
    DOT_PROD = 0x3a2fd47a
    ADD = 0x5135fabc
    NOP = 0x3823ed06
    SLEEP = 0x39d768e9
    ADD_LATER = 0x64e0a594
    LOSSY_NOP = 0x5323ea07
    FAST_LOSSY_NOP = 0x6223e760

    __input_type_info__ = {
        'fast_prime': ['rpc::i32'],
        'fast_dot_prod': ['point3','point3'],
        'fast_add': ['rpc::v32','rpc::v32'],
        'fast_nop': ['std::string'],
        'prime': ['rpc::i32'],
        'dot_prod': ['point3','point3'],
        'add': ['rpc::v32','rpc::v32'],
        'nop': ['std::string'],
        'sleep': ['double'],
        'add_later': ['rpc::i32','rpc::i32'],
        'lossy_nop': ['rpc::i32','rpc::i32'],
        'fast_lossy_nop': [],
    }

    __output_type_info__ = {
        'fast_prime': ['rpc::i8'],
        'fast_dot_prod': ['double'],
        'fast_add': ['rpc::v32'],
        'fast_nop': [],
        'prime': ['rpc::i8'],
        'dot_prod': ['double'],
        'add': ['rpc::v32'],
        'nop': [],
        'sleep': [],
        'add_later': ['rpc::i32'],
        'lossy_nop': [],
        'fast_lossy_nop': [],
    }

    def __bind_helper__(self, func):
        def f(*args):
            return getattr(self, func.__name__)(*args)
        return f

    def __reg_to__(self, server):
        server.__reg_func__(BenchmarkService.FAST_PRIME, self.__bind_helper__(self.fast_prime), ['rpc::i32'], ['rpc::i8'])
        server.__reg_func__(BenchmarkService.FAST_DOT_PROD, self.__bind_helper__(self.fast_dot_prod), ['point3','point3'], ['double'])
        server.__reg_func__(BenchmarkService.FAST_ADD, self.__bind_helper__(self.fast_add), ['rpc::v32','rpc::v32'], ['rpc::v32'])
        server.__reg_func__(BenchmarkService.FAST_NOP, self.__bind_helper__(self.fast_nop), ['std::string'], [])
        server.__reg_func__(BenchmarkService.PRIME, self.__bind_helper__(self.prime), ['rpc::i32'], ['rpc::i8'])
        server.__reg_func__(BenchmarkService.DOT_PROD, self.__bind_helper__(self.dot_prod), ['point3','point3'], ['double'])
        server.__reg_func__(BenchmarkService.ADD, self.__bind_helper__(self.add), ['rpc::v32','rpc::v32'], ['rpc::v32'])
        server.__reg_func__(BenchmarkService.NOP, self.__bind_helper__(self.nop), ['std::string'], [])
        server.__reg_func__(BenchmarkService.SLEEP, self.__bind_helper__(self.sleep), ['double'], [])
        server.__reg_func__(BenchmarkService.ADD_LATER, self.__bind_helper__(self.add_later), ['rpc::i32','rpc::i32'], ['rpc::i32'])
        server.enable_udp()
        server.__reg_func__(BenchmarkService.LOSSY_NOP, self.__bind_helper__(self.lossy_nop), ['rpc::i32','rpc::i32'], [])
        server.__reg_func__(BenchmarkService.FAST_LOSSY_NOP, self.__bind_helper__(self.fast_lossy_nop), [], [])

    def fast_prime(__self__, n):
        raise NotImplementedError('subclass BenchmarkService and implement your own fast_prime function')

    def fast_dot_prod(__self__, p1, p2):
        raise NotImplementedError('subclass BenchmarkService and implement your own fast_dot_prod function')

    def fast_add(__self__, a, b):
        raise NotImplementedError('subclass BenchmarkService and implement your own fast_add function')

    def fast_nop(__self__, in0):
        raise NotImplementedError('subclass BenchmarkService and implement your own fast_nop function')

    def prime(__self__, n):
        raise NotImplementedError('subclass BenchmarkService and implement your own prime function')

    def dot_prod(__self__, p1, p2):
        raise NotImplementedError('subclass BenchmarkService and implement your own dot_prod function')

    def add(__self__, a, b):
        raise NotImplementedError('subclass BenchmarkService and implement your own add function')

    def nop(__self__, in0):
        raise NotImplementedError('subclass BenchmarkService and implement your own nop function')

    def sleep(__self__, sec):
        raise NotImplementedError('subclass BenchmarkService and implement your own sleep function')

    def add_later(__self__, a, b):
        raise NotImplementedError('subclass BenchmarkService and implement your own add_later function')

    def lossy_nop(__self__, dummy, dummy2):
        raise NotImplementedError('subclass BenchmarkService and implement your own lossy_nop function')

    def fast_lossy_nop(__self__):
        raise NotImplementedError('subclass BenchmarkService and implement your own fast_lossy_nop function')

class BenchmarkProxy(object):
    def __init__(self, clnt):
        self.__clnt__ = clnt

    def async_fast_prime(__self__, n):
        return __self__.__clnt__.async_call(BenchmarkService.FAST_PRIME, [n], BenchmarkService.__input_type_info__['fast_prime'], BenchmarkService.__output_type_info__['fast_prime'])

    def async_fast_dot_prod(__self__, p1, p2):
        return __self__.__clnt__.async_call(BenchmarkService.FAST_DOT_PROD, [p1, p2], BenchmarkService.__input_type_info__['fast_dot_prod'], BenchmarkService.__output_type_info__['fast_dot_prod'])

    def async_fast_add(__self__, a, b):
        return __self__.__clnt__.async_call(BenchmarkService.FAST_ADD, [a, b], BenchmarkService.__input_type_info__['fast_add'], BenchmarkService.__output_type_info__['fast_add'])

    def async_fast_nop(__self__, in0):
        return __self__.__clnt__.async_call(BenchmarkService.FAST_NOP, [in0], BenchmarkService.__input_type_info__['fast_nop'], BenchmarkService.__output_type_info__['fast_nop'])

    def async_prime(__self__, n):
        return __self__.__clnt__.async_call(BenchmarkService.PRIME, [n], BenchmarkService.__input_type_info__['prime'], BenchmarkService.__output_type_info__['prime'])

    def async_dot_prod(__self__, p1, p2):
        return __self__.__clnt__.async_call(BenchmarkService.DOT_PROD, [p1, p2], BenchmarkService.__input_type_info__['dot_prod'], BenchmarkService.__output_type_info__['dot_prod'])

    def async_add(__self__, a, b):
        return __self__.__clnt__.async_call(BenchmarkService.ADD, [a, b], BenchmarkService.__input_type_info__['add'], BenchmarkService.__output_type_info__['add'])

    def async_nop(__self__, in0):
        return __self__.__clnt__.async_call(BenchmarkService.NOP, [in0], BenchmarkService.__input_type_info__['nop'], BenchmarkService.__output_type_info__['nop'])

    def async_sleep(__self__, sec):
        return __self__.__clnt__.async_call(BenchmarkService.SLEEP, [sec], BenchmarkService.__input_type_info__['sleep'], BenchmarkService.__output_type_info__['sleep'])

    def async_add_later(__self__, a, b):
        return __self__.__clnt__.async_call(BenchmarkService.ADD_LATER, [a, b], BenchmarkService.__input_type_info__['add_later'], BenchmarkService.__output_type_info__['add_later'])

    def sync_fast_prime(__self__, n):
        __result__ = __self__.__clnt__.sync_call(BenchmarkService.FAST_PRIME, [n], BenchmarkService.__input_type_info__['fast_prime'], BenchmarkService.__output_type_info__['fast_prime'])
        if __result__[0] != 0:
            raise Exception("RPC returned non-zero error code %d: %s" % (__result__[0], os.strerror(__result__[0])))
        if len(__result__[1]) == 1:
            return __result__[1][0]
        elif len(__result__[1]) > 1:
            return __result__[1]

    def sync_fast_dot_prod(__self__, p1, p2):
        __result__ = __self__.__clnt__.sync_call(BenchmarkService.FAST_DOT_PROD, [p1, p2], BenchmarkService.__input_type_info__['fast_dot_prod'], BenchmarkService.__output_type_info__['fast_dot_prod'])
        if __result__[0] != 0:
            raise Exception("RPC returned non-zero error code %d: %s" % (__result__[0], os.strerror(__result__[0])))
        if len(__result__[1]) == 1:
            return __result__[1][0]
        elif len(__result__[1]) > 1:
            return __result__[1]

    def sync_fast_add(__self__, a, b):
        __result__ = __self__.__clnt__.sync_call(BenchmarkService.FAST_ADD, [a, b], BenchmarkService.__input_type_info__['fast_add'], BenchmarkService.__output_type_info__['fast_add'])
        if __result__[0] != 0:
            raise Exception("RPC returned non-zero error code %d: %s" % (__result__[0], os.strerror(__result__[0])))
        if len(__result__[1]) == 1:
            return __result__[1][0]
        elif len(__result__[1]) > 1:
            return __result__[1]

    def sync_fast_nop(__self__, in0):
        __result__ = __self__.__clnt__.sync_call(BenchmarkService.FAST_NOP, [in0], BenchmarkService.__input_type_info__['fast_nop'], BenchmarkService.__output_type_info__['fast_nop'])
        if __result__[0] != 0:
            raise Exception("RPC returned non-zero error code %d: %s" % (__result__[0], os.strerror(__result__[0])))
        if len(__result__[1]) == 1:
            return __result__[1][0]
        elif len(__result__[1]) > 1:
            return __result__[1]

    def sync_prime(__self__, n):
        __result__ = __self__.__clnt__.sync_call(BenchmarkService.PRIME, [n], BenchmarkService.__input_type_info__['prime'], BenchmarkService.__output_type_info__['prime'])
        if __result__[0] != 0:
            raise Exception("RPC returned non-zero error code %d: %s" % (__result__[0], os.strerror(__result__[0])))
        if len(__result__[1]) == 1:
            return __result__[1][0]
        elif len(__result__[1]) > 1:
            return __result__[1]

    def sync_dot_prod(__self__, p1, p2):
        __result__ = __self__.__clnt__.sync_call(BenchmarkService.DOT_PROD, [p1, p2], BenchmarkService.__input_type_info__['dot_prod'], BenchmarkService.__output_type_info__['dot_prod'])
        if __result__[0] != 0:
            raise Exception("RPC returned non-zero error code %d: %s" % (__result__[0], os.strerror(__result__[0])))
        if len(__result__[1]) == 1:
            return __result__[1][0]
        elif len(__result__[1]) > 1:
            return __result__[1]

    def sync_add(__self__, a, b):
        __result__ = __self__.__clnt__.sync_call(BenchmarkService.ADD, [a, b], BenchmarkService.__input_type_info__['add'], BenchmarkService.__output_type_info__['add'])
        if __result__[0] != 0:
            raise Exception("RPC returned non-zero error code %d: %s" % (__result__[0], os.strerror(__result__[0])))
        if len(__result__[1]) == 1:
            return __result__[1][0]
        elif len(__result__[1]) > 1:
            return __result__[1]

    def sync_nop(__self__, in0):
        __result__ = __self__.__clnt__.sync_call(BenchmarkService.NOP, [in0], BenchmarkService.__input_type_info__['nop'], BenchmarkService.__output_type_info__['nop'])
        if __result__[0] != 0:
            raise Exception("RPC returned non-zero error code %d: %s" % (__result__[0], os.strerror(__result__[0])))
        if len(__result__[1]) == 1:
            return __result__[1][0]
        elif len(__result__[1]) > 1:
            return __result__[1]

    def sync_sleep(__self__, sec):
        __result__ = __self__.__clnt__.sync_call(BenchmarkService.SLEEP, [sec], BenchmarkService.__input_type_info__['sleep'], BenchmarkService.__output_type_info__['sleep'])
        if __result__[0] != 0:
            raise Exception("RPC returned non-zero error code %d: %s" % (__result__[0], os.strerror(__result__[0])))
        if len(__result__[1]) == 1:
            return __result__[1][0]
        elif len(__result__[1]) > 1:
            return __result__[1]

    def sync_add_later(__self__, a, b):
        __result__ = __self__.__clnt__.sync_call(BenchmarkService.ADD_LATER, [a, b], BenchmarkService.__input_type_info__['add_later'], BenchmarkService.__output_type_info__['add_later'])
        if __result__[0] != 0:
            raise Exception("RPC returned non-zero error code %d: %s" % (__result__[0], os.strerror(__result__[0])))
        if len(__result__[1]) == 1:
            return __result__[1][0]
        elif len(__result__[1]) > 1:
            return __result__[1]

    def udp_lossy_nop(__self__, dummy, dummy2):
        return __self__.__clnt__.udp_call(BenchmarkService.LOSSY_NOP, [dummy, dummy2], BenchmarkService.__input_type_info__['lossy_nop'])

    def udp_fast_lossy_nop(__self__):
        return __self__.__clnt__.udp_call(BenchmarkService.FAST_LOSSY_NOP, [], BenchmarkService.__input_type_info__['fast_lossy_nop'])

