from simplerpcgen.misc import SourceFile

def emit_struct(struct, f):
    f.writeln("struct %s {" % struct.name)
    with f.indent():
        for field in struct.fields:
            f.writeln("%s %s;" % (field.type, field.name))
    f.writeln("};")
    f.writeln()
    f.writeln("inline rpc::Marshal& operator <<(rpc::Marshal& m, const %s& o) {" % struct.name)
    with f.indent():
        for field in struct.fields:
            f.writeln("m << o.%s;" % field.name)
        f.writeln("return m;")
    f.writeln("}")
    f.writeln()
    f.writeln("inline rpc::Marshal& operator >>(rpc::Marshal& m, %s& o) {" % struct.name)
    with f.indent():
        for field in struct.fields:
            f.writeln("m >> o.%s;" % field.name)
        f.writeln("return m;")
    f.writeln("}")
    f.writeln()

def emit_service_and_proxy(service, f, rpc_table):
    f.writeln("class %sService: public rpc::Service {" % service.name)
    f.writeln("public:")
    with f.indent():
        f.writeln("enum {")
        with f.indent():
            for func in service.functions:
                rpc_code = rpc_table["%s.%s" % (service.name, func.name)]
                f.writeln("%s = %s," % (func.name.upper(), hex(rpc_code)))
        f.writeln("};")
        udp_enabled = False
        f.writeln("int __reg_to__(rpc::Server* svr) {")
        with f.indent():
            f.writeln("int ret = 0;")
            for func in service.functions:
                if "udp" in func.attrs and not udp_enabled:
                    f.writeln("svr->enable_udp();")
                    udp_enabled = True
                if "raw" in func.attrs:
                    f.writeln("if ((ret = svr->reg(%s, this, &%sService::%s)) != 0) {" % (func.name.upper(), service.name, func.name))
                else:
                    f.writeln("if ((ret = svr->reg(%s, this, &%sService::__%s__wrapper__)) != 0) {" % (func.name.upper(), service.name, func.name))
                with f.indent():
                    f.writeln("goto err;")
                f.writeln("}")
            f.writeln("return 0;")
        f.writeln("err:")
        with f.indent():
            for func in service.functions:
                f.writeln("svr->unreg(%s);" % func.name.upper())
            f.writeln("return ret;")
        f.writeln("}")
        f.writeln("// these RPC handler functions need to be implemented by user")
        f.writeln("// for 'raw' handlers, remember to reply req, delete req, and sconn->release(); use sconn->run_async for heavy job")
        for func in service.functions:
            if service.abstract or func.abstract:
                postfix = " = 0"
            else:
                postfix = ""
            if "raw" in func.attrs:
                f.writeln("virtual void %s(rpc::Request* req, rpc::ServerConnection* sconn)%s;" % (func.name, postfix))
            else:
                func_args = []
                for in_arg in func.input:
                    if in_arg.name != None:
                        func_args += "const %s& %s" % (in_arg.type, in_arg.name),
                    else:
                        func_args += "const %s&" % in_arg.type,
                for out_arg in func.output:
                    if out_arg.name != None:
                        func_args += "%s* %s" % (out_arg.type, out_arg.name),
                    else:
                        func_args += "%s*" % out_arg.type,
                if "defer" in func.attrs:
                    func_args += "rpc::DeferredReply* defer",
                f.writeln("virtual void %s(%s)%s;" % (func.name, ", ".join(func_args), postfix))
    f.writeln("private:")
    with f.indent():
        for func in service.functions:
            if "raw" in func.attrs:
                continue
            f.writeln("void __%s__wrapper__(rpc::Request* req, rpc::ServerConnection* sconn) {" % func.name)
            with f.indent():
                if "defer" in func.attrs:
                    invoke_with = []
                    in_counter = 0
                    out_counter = 0
                    for in_arg in func.input:
                        f.writeln("%s* in_%d = new %s;" % (in_arg.type, in_counter, in_arg.type))
                        f.writeln("req->m >> *in_%d;" % in_counter)
                        invoke_with += "*in_%d" % in_counter,
                        in_counter += 1
                    for out_arg in func.output:
                        f.writeln("%s* out_%d = new %s;" % (out_arg.type, out_counter, out_arg.type))
                        invoke_with += "out_%d" % out_counter,
                        out_counter += 1
                    f.writeln("auto __marshal_reply__ = [=] {");
                    with f.indent():
                        out_counter = 0
                        for out_arg in func.output:
                            f.writeln("*sconn << *out_%d;" % out_counter)
                            out_counter += 1
                    f.writeln("};");
                    f.writeln("auto __cleanup__ = [=] {");
                    with f.indent():
                        in_counter = 0
                        out_counter = 0
                        for in_arg in func.input:
                            f.writeln("delete in_%d;" % in_counter)
                            in_counter += 1
                        for out_arg in func.output:
                            f.writeln("delete out_%d;" % out_counter)
                            out_counter += 1
                    f.writeln("};");
                    f.writeln("rpc::DeferredReply* __defer__ = new rpc::DeferredReply(req, sconn, __marshal_reply__, __cleanup__);")
                    invoke_with += "__defer__",
                    f.writeln("this->%s(%s);" % (func.name, ", ".join(invoke_with)))
                else: # normal and fast rpc
                    if "fast" not in func.attrs:
                        f.writeln("auto f = [=] {")
                        f.incr_indent()
                    invoke_with = []
                    in_counter = 0
                    out_counter = 0
                    for in_arg in func.input:
                        f.writeln("%s in_%d;" % (in_arg.type, in_counter))
                        f.writeln("req->m >> in_%d;" % in_counter)
                        invoke_with += "in_%d" % in_counter,
                        in_counter += 1
                    for out_arg in func.output:
                        f.writeln("%s out_%d;" % (out_arg.type, out_counter))
                        invoke_with += "&out_%d" % out_counter,
                        out_counter += 1
                    f.writeln("this->%s(%s);" % (func.name, ", ".join(invoke_with)))
                    if "udp" not in func.attrs:
                        f.writeln("sconn->begin_reply(req);")
                        for i in range(out_counter):
                            f.writeln("*sconn << out_%d;" % i)
                        f.writeln("sconn->end_reply();")
                    f.writeln("delete req;")
                    f.writeln("sconn->release();")
                    if "fast" not in func.attrs:
                        f.decr_indent()
                        f.writeln("};")
                        f.writeln("sconn->run_async(f);")
            f.writeln("}")
    f.writeln("};")
    f.writeln()
    f.writeln("class %sProxy {" % service.name)
    f.writeln("protected:")
    with f.indent():
        f.writeln("rpc::Client* __cl__;")
    f.writeln("public:")
    with f.indent():
        f.writeln("%sProxy(rpc::Client* cl): __cl__(cl) { }" % service.name)
        for func in service.functions:
            async_func_params = []
            async_call_params = []
            sync_func_params = []
            sync_out_params = []
            in_counter = 0
            out_counter = 0
            for in_arg in func.input:
                if in_arg.name != None:
                    async_func_params += "const %s& %s" % (in_arg.type, in_arg.name),
                    async_call_params += in_arg.name,
                    sync_func_params += "const %s& %s" % (in_arg.type, in_arg.name),
                else:
                    async_func_params += "const %s& in_%d" % (in_arg.type, in_counter),
                    async_call_params += "in_%d" % in_counter,
                    sync_func_params += "const %s& in_%d" % (in_arg.type, in_counter),
                in_counter += 1
            for out_arg in func.output:
                if out_arg.name != None:
                    sync_func_params += "%s* %s" % (out_arg.type, out_arg.name),
                    sync_out_params += out_arg.name,
                else:
                    sync_func_params += "%s* out_%d" % (out_arg.type, out_counter),
                    sync_out_params += "out_%d" % out_counter,
                out_counter += 1

            if "udp" in func.attrs:
                f.writeln("int %s(%s) /* UDP */ {" % (func.name, ", ".join(sync_func_params)))
                with f.indent():
                    f.writeln("__cl__->begin_udp_request(%sService::%s);" % (service.name, func.name.upper()))
                    for param in async_call_params:
                        f.writeln("__cl__->udp_request() << %s;" % param)
                    f.writeln("return __cl__->end_udp_request();")
                f.writeln("}")
                continue

            f.writeln("rpc::Future* async_%s(%sconst rpc::FutureAttr& __fu_attr__ = rpc::FutureAttr()) {" % (func.name, ", ".join(async_func_params + [""])))
            with f.indent():
                f.writeln("rpc::Future* __fu__ = __cl__->begin_request(%sService::%s, __fu_attr__);" % (service.name, func.name.upper()))
                if len(async_call_params) > 0:
                    f.writeln("if (__fu__ != nullptr) {")
                    with f.indent():
                        for param in async_call_params:
                            f.writeln("*__cl__ << %s;" % param)
                    f.writeln("}")
                f.writeln("__cl__->end_request();")
                f.writeln("return __fu__;")
            f.writeln("}")
            f.writeln("rpc::i32 %s(%s) {" % (func.name, ", ".join(sync_func_params)))
            with f.indent():
                f.writeln("rpc::Future* __fu__ = this->async_%s(%s);" % (func.name, ", ".join(async_call_params)))
                f.writeln("if (__fu__ == nullptr) {")
                with f.indent():
                    f.writeln("return ENOTCONN;")
                f.writeln("}")
                f.writeln("rpc::i32 __ret__ = __fu__->get_error_code();")
                if len(sync_out_params) > 0:
                    f.writeln("if (__ret__ == 0) {")
                    with f.indent():
                        for param in sync_out_params:
                            f.writeln("__fu__->get_reply() >> *%s;" % param)
                    f.writeln("}")
                f.writeln("__fu__->release();")
                f.writeln("return __ret__;")
            f.writeln("}")
    f.writeln("};")
    f.writeln()


def emit_rpc_source_cpp(rpc_source, rpc_table, fpath, cpp_header, cpp_footer):
    with open(fpath, "w") as f:
        f = SourceFile(f)
        f.writeln("#pragma once")
        f.writeln()
        f.writeln('#include "rpc/server.h"')
        f.writeln('#include "rpc/client.h"')
        f.writeln()
        f.writeln("#include <errno.h>")
        f.writeln()
        f.write(cpp_header)
        f.writeln()

        if rpc_source.namespace != None:
            f.writeln(" ".join(map(lambda x:"namespace %s {" % x, rpc_source.namespace)))
            f.writeln()

        for struct in rpc_source.structs:
            emit_struct(struct, f)

        for service in rpc_source.services:
            emit_service_and_proxy(service, f, rpc_table)

        if rpc_source.namespace != None:
            f.writeln(" ".join(["}"] * len(rpc_source.namespace)) + " // namespace " + "::".join(rpc_source.namespace))
            f.writeln()

        f.writeln()
        f.write(cpp_footer)
        f.writeln()
