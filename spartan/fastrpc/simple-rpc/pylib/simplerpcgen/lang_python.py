from simplerpcgen.misc import SourceFile

def emit_struct_python(struct, f):
    f.writeln("%s = Marshal.reg_type('%s', [%s])" % (
        struct.name, struct.name, ", ".join(["('%s', '%s')" % (field.name, field.type) for field in struct.fields])))
    f.writeln()


def emit_service_and_proxy_python(service, f, rpc_table):
    f.writeln("class %sService(object):" % service.name)
    with f.indent():
        for func in service.functions:
            rpc_id = rpc_table["%s.%s" % (service.name, func.name)]
            f.writeln("%s = %s" % (func.name.upper(), hex(rpc_id)))
        f.writeln()
        f.writeln("__input_type_info__ = {")
        with f.indent():
            for func in service.functions:
                f.writeln("'%s': [%s]," % (func.name, ",".join(["'%s'" % a.type for a in func.input])))
        f.writeln("}")
        f.writeln()
        f.writeln("__output_type_info__ = {")
        with f.indent():
            for func in service.functions:
                f.writeln("'%s': [%s]," % (func.name, ",".join(["'%s'" % a.type for a in func.output])))
        f.writeln("}")
        f.writeln()
        f.writeln("def __bind_helper__(self, func):")
        with f.indent():
            f.writeln("def f(*args):")
            with f.indent():
                f.writeln("return getattr(self, func.__name__)(*args)")
            f.writeln("return f")
        f.writeln()
        f.writeln("def __reg_to__(self, server):")
        with f.indent():
            udp_enabled = False
            for func in service.functions:
                if "udp" in func.attrs and not udp_enabled:
                    f.writeln("server.enable_udp()")
                    udp_enabled = True
                f.writeln("server.__reg_func__(%sService.%s, self.__bind_helper__(self.%s), [%s], [%s])" % (
                    service.name, func.name.upper(), func.name,
                    ",".join(["'%s'" % a.type for a in func.input]),
                    ",".join(["'%s'" % a.type for a in func.output])))
            if len(service.functions) == 0:
                f.writeln("pass")
        for func in service.functions:
            f.writeln()
            in_params_decl = ""
            for i in range(len(func.input)):
                in_arg = func.input[i]
                if in_arg.name != None:
                    in_params_decl += ", " + in_arg.name
                else:
                    in_params_decl += ", in%d" % i
            f.writeln("def %s(__self__%s):" % (func.name, in_params_decl))
            with f.indent():
                f.writeln("raise NotImplementedError('subclass %sService and implement your own %s function')" % (service.name, func.name))
    f.writeln()

    f.writeln("class %sProxy(object):" % service.name)
    with f.indent():
        f.writeln("def __init__(self, clnt):")
        with f.indent():
            f.writeln("self.__clnt__ = clnt")

        for func in service.functions:
            if "udp" in func.attrs:
                continue
            f.writeln()
            in_params_decl = ""
            for i in range(len(func.input)):
                in_arg = func.input[i]
                if in_arg.name != None:
                    in_params_decl += ", " + in_arg.name
                else:
                    in_params_decl += ", in%d" % i

            f.writeln("def async_%s(__self__%s):" % (func.name, in_params_decl))
            with f.indent():
                if in_params_decl != "":
                    in_params_decl = in_params_decl[2:] # strip the leading ", "
                f.writeln("return __self__.__clnt__.async_call(%sService.%s, [%s], %sService.__input_type_info__['%s'], %sService.__output_type_info__['%s'])" % (
                    service.name, func.name.upper(), in_params_decl, service.name, func.name, service.name, func.name))

        for func in service.functions:
            if "udp" in func.attrs:
                continue
            f.writeln()
            in_params_decl = ""
            for i in range(len(func.input)):
                in_arg = func.input[i]
                if in_arg.name != None:
                    in_params_decl += ", " + in_arg.name
                else:
                    in_params_decl += ", in%d" % i

            f.writeln("def sync_%s(__self__%s):" % (func.name, in_params_decl))
            with f.indent():
                if in_params_decl != "":
                    in_params_decl = in_params_decl[2:] # strip the leading ", "
                f.writeln("__result__ = __self__.__clnt__.sync_call(%sService.%s, [%s], %sService.__input_type_info__['%s'], %sService.__output_type_info__['%s'])" % (
                    service.name, func.name.upper(), in_params_decl, service.name, func.name, service.name, func.name))
                f.writeln("if __result__[0] != 0:")
                with f.indent():
                    f.writeln("raise Exception(\"RPC returned non-zero error code %d: %s\" % (__result__[0], os.strerror(__result__[0])))")
                f.writeln("if len(__result__[1]) == 1:")
                with f.indent():
                    f.writeln("return __result__[1][0]")
                f.writeln("elif len(__result__[1]) > 1:")
                with f.indent():
                    f.writeln("return __result__[1]")

        for func in service.functions:
            if "udp" not in func.attrs:
                continue
            f.writeln()
            in_params_decl = ""
            for i in range(len(func.input)):
                in_arg = func.input[i]
                if in_arg.name != None:
                    in_params_decl += ", " + in_arg.name
                else:
                    in_params_decl += ", in%d" % i

            f.writeln("def udp_%s(__self__%s):" % (func.name, in_params_decl))
            with f.indent():
                if in_params_decl != "":
                    in_params_decl = in_params_decl[2:] # strip the leading ", "
                f.writeln("return __self__.__clnt__.udp_call(%sService.%s, [%s], %sService.__input_type_info__['%s'])" % (
                    service.name, func.name.upper(), in_params_decl, service.name, func.name))

    f.writeln()


def emit_rpc_source_python(rpc_source, rpc_table, fpath):
    with open(fpath, "w") as f:
        f = SourceFile(f)
        f.writeln("import os")
        f.writeln("from simplerpc.marshal import Marshal")
        f.writeln("from simplerpc.future import Future")
        f.writeln()
        for struct in rpc_source.structs:
            emit_struct_python(struct, f)
        for service in rpc_source.services:
            emit_service_and_proxy_python(service, f, rpc_table)
