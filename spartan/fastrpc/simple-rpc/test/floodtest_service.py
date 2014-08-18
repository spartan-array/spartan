import os
from simplerpc.marshal import Marshal
from simplerpc.future import Future

class FloodService(object):
    UPDATE_NODE_LIST = 0x43102c1b
    FLOOD = 0x4c64a660
    FLOOD_UDP = 0x632a2de2

    __input_type_info__ = {
        'update_node_list': ['std::vector<std::string>'],
        'flood': [],
        'flood_udp': [],
    }

    __output_type_info__ = {
        'update_node_list': [],
        'flood': [],
        'flood_udp': [],
    }

    def __bind_helper__(self, func):
        def f(*args):
            return getattr(self, func.__name__)(*args)
        return f

    def __reg_to__(self, server):
        server.__reg_func__(FloodService.UPDATE_NODE_LIST, self.__bind_helper__(self.update_node_list), ['std::vector<std::string>'], [])
        server.__reg_func__(FloodService.FLOOD, self.__bind_helper__(self.flood), [], [])
        server.enable_udp()
        server.__reg_func__(FloodService.FLOOD_UDP, self.__bind_helper__(self.flood_udp), [], [])

    def update_node_list(__self__, nodes):
        raise NotImplementedError('subclass FloodService and implement your own update_node_list function')

    def flood(__self__):
        raise NotImplementedError('subclass FloodService and implement your own flood function')

    def flood_udp(__self__):
        raise NotImplementedError('subclass FloodService and implement your own flood_udp function')

class FloodProxy(object):
    def __init__(self, clnt):
        self.__clnt__ = clnt

    def async_update_node_list(__self__, nodes):
        return __self__.__clnt__.async_call(FloodService.UPDATE_NODE_LIST, [nodes], FloodService.__input_type_info__['update_node_list'], FloodService.__output_type_info__['update_node_list'])

    def async_flood(__self__):
        return __self__.__clnt__.async_call(FloodService.FLOOD, [], FloodService.__input_type_info__['flood'], FloodService.__output_type_info__['flood'])

    def sync_update_node_list(__self__, nodes):
        __result__ = __self__.__clnt__.sync_call(FloodService.UPDATE_NODE_LIST, [nodes], FloodService.__input_type_info__['update_node_list'], FloodService.__output_type_info__['update_node_list'])
        if __result__[0] != 0:
            raise Exception("RPC returned non-zero error code %d: %s" % (__result__[0], os.strerror(__result__[0])))
        if len(__result__[1]) == 1:
            return __result__[1][0]
        elif len(__result__[1]) > 1:
            return __result__[1]

    def sync_flood(__self__):
        __result__ = __self__.__clnt__.sync_call(FloodService.FLOOD, [], FloodService.__input_type_info__['flood'], FloodService.__output_type_info__['flood'])
        if __result__[0] != 0:
            raise Exception("RPC returned non-zero error code %d: %s" % (__result__[0], os.strerror(__result__[0])))
        if len(__result__[1]) == 1:
            return __result__[1][0]
        elif len(__result__[1]) > 1:
            return __result__[1]

    def udp_flood_udp(__self__):
        return __self__.__clnt__.udp_call(FloodService.FLOOD_UDP, [], FloodService.__input_type_info__['flood_udp'])

