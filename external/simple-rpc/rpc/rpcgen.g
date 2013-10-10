#!/usr/bin/env python

import sys
import os
import random
import re
sys.path += os.path.abspath(os.path.join(os.path.split(__file__)[0], "../pylib")),


def error(msg, ctx):
    from yapps import runtime
    err = runtime.SyntaxError(None, msg, ctx)
    runtime.print_error(err, ctx.scanner)
    sys.exit(1)


class pack:
    def __init__(self, **kv):
        self.__dict__.update(kv)
    def __str__(self):
        return str(self.__dict__)

def std_rename(t):
    if t in ["string", "map", "list", "set", "deque", "vector", "unordered_map", "unordered_set"]:
        t = "std::" + t
    return t

def forbid_reserved_names(name):
    if re.match("__([^_]+.*[^_]+|[^_])__$", name):
        raise Exception("bad name '%s', __NAME__ format names are reserved" % name)

%%

parser Rpc:

    ignore:     "\\s+"              # ignore space
    ignore:     "//[^\\n]+"         # ignore comment
    ignore:     ";"                 # ignore end-of-line ;

    token EOF:      "($|%%)"      # %% marks begining of "raw copying source code"
    token SYMBOL:   "[a-zA-Z_][a-zA-Z0-9_]*"

    rule rpc_source: {{ namespace = None }}
        [namespace_decl {{ namespace = namespace_decl }}] structs_and_services EOF
            {{ return pack(namespace=namespace, structs=structs_and_services.structs, services=structs_and_services.services) }}

    rule namespace_decl:
        "namespace" SYMBOL {{ namespace = [SYMBOL] }} ("::" SYMBOL {{ namespace += SYMBOL, }})*
            {{ return namespace }}

    rule structs_and_services: {{ structs = []; services = [] }}
        (struct_decl {{ structs += struct_decl, }} | service_decl {{ services += service_decl, }})*
            {{ return pack(structs=structs, services=services) }}

    rule struct_decl:
        "struct" SYMBOL "{" struct_fields "}"
            {{ return pack(name=SYMBOL, fields=struct_fields) }}

    rule struct_fields: {{ fields = [] }}
        (struct_field {{ fields += struct_field, }})*
            {{ return fields }}

    rule struct_field:
        type SYMBOL
            {{ return pack(name=SYMBOL, type=type) }}

    rule type:
        "i32" {{ return "rpc::i32" }}
        | "i64" {{ return "rpc::i64" }}
        | full_symbol {{ t = std_rename(full_symbol) }}
            ["<" type {{ t += "<" + type }} ("," type {{ t += ", " + type }})* ">" {{ t += ">" }}] {{ return t }}
        | ("bool" | "int" | "unsigned" | "long" ) {{ error("please use i32 or i64 for any integer types", _context) }}

    rule full_symbol: {{ s = "" }}
        ["::" {{ s += "::" }}] SYMBOL {{ s += SYMBOL }} ("::" SYMBOL {{ s += "::" + SYMBOL }})*
            {{ return s }}

    rule service_decl: {{ abstract = False }}
        ["abstract" {{ abstract = True }}] "service" SYMBOL "{" service_functions "}"
            {{ return pack(name=SYMBOL, abstract=abstract, functions=service_functions) }}

    rule service_functions: {{ functions = [] }}
        (service_function {{ functions += service_function, }})*
            {{ return functions }}

    rule service_function: {{ attr = None; abstract = False; input = []; output = [] }}
        ["fast" {{ attr = "fast" }} | "raw" {{ attr = "raw" }}]
        SYMBOL {{ forbid_reserved_names(SYMBOL) }}
        "\(" (func_arg_list {{ input = func_arg_list }}) ["\|" (func_arg_list {{ output = func_arg_list }})] "\)"
        ["=" "0" {{ abstract = True }}]
            {{ return pack(name=SYMBOL, attr=attr, abstract=abstract, input=input, output=output) }}

    rule func_arg_list: {{ args = [] }}
        (| func_arg {{ args = [func_arg] }} ("," func_arg {{ args += func_arg, }})*)
            {{ return args }}

    rule func_arg: {{ name = None }}
        type [SYMBOL {{ name = SYMBOL; forbid_reserved_names(name) }}]
            {{ return pack(name=name, type=type) }}

%%

class SourceFile(object):
    def __init__(self, f):
        self.f = f
        self.indent_level = 0
    def indent(self):
        class Indent:
            def __init__(self, sf):
                self.sf = sf
            def __enter__(self):
                self.sf.indent_level += 1
            def __exit__(self, type, value, traceback):
                self.sf.indent_level -= 1
        return Indent(self)
    def incr_indent(self):
        self.indent_level += 1
    def decr_indent(self):
        self.indent_level -= 1
        assert self.indent_level >= 0
        
    def write(self, txt):
        self.f.write(txt)
        
    def writeln(self, txt=None):
        if txt != None:
            self.f.write("    " * self.indent_level)
            self.f.write(txt)
        self.f.write("\n")

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


def emit_service_and_proxy(service, f):
    f.writeln("class %sService: public rpc::Service {" % service.name)
    f.writeln("public:")
    with f.indent():
        f.writeln("enum {")
        with f.indent():
            for func in service.functions:
                rpc_code = random.randint(0x10000000, 0x70000000)
                f.writeln("%s = %s," % (func.name.upper(), hex(rpc_code)))
        f.writeln("};")
        f.writeln("int reg_to(rpc::Server* svr) {")
        with f.indent():
            f.writeln("int ret = 0;")
            for func in service.functions:
                if func.attr == "raw":
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
            if func.attr == "raw":
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
                f.writeln("virtual void %s(%s)%s;" % (func.name, ", ".join(func_args), postfix))
    f.writeln("private:")
    with f.indent():
        for func in service.functions:
            if func.attr == "raw":
                continue
            f.writeln("void __%s__wrapper__(rpc::Request* req, rpc::ServerConnection* sconn) {" % func.name)
            with f.indent():
                if func.attr != "fast":
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
                f.writeln("sconn->begin_reply(req);")
                for i in range(out_counter):
                    f.writeln("*sconn << out_%d;" % i)
                f.writeln("sconn->end_reply();")
                f.writeln("delete req;")
                f.writeln("sconn->release();")
                if func.attr != "fast":
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


def emit_rpc_source(rpc_source, f):
    if rpc_source.namespace != None:
        f.writeln(" ".join(map(lambda x:"namespace %s {" % x, rpc_source.namespace)))
        f.writeln()

    for struct in rpc_source.structs:
        emit_struct(struct, f)

    for service in rpc_source.services:
        emit_service_and_proxy(service, f)

    if rpc_source.namespace != None:
        f.writeln(" ".join(["}"] * len(rpc_source.namespace)) + " // namespace " + "::".join(rpc_source.namespace))
        f.writeln()

def rpcgen(rpc_fpath):
    with open(rpc_fpath) as f:
        rpc_src = f.read()
        
    rpc_src_lines = [l.strip() for l in rpc_src.split("\n")]
   
    header = footer = src = ''
    
    if rpc_src_lines.count('%%') == 2:
	    # header + source + footer
	    first = rpc_src_lines.index("%%")
	    next = rpc_src_lines.index("%%", first + 1)
	    header =  '\n'.join(rpc_src_lines[:first])
	    src = '\n'.join(rpc_src_lines[first+1:next])
	    footer = '\n'.join(rpc_src_lines[next + 1:])
    elif rpc_src_lines.count('%%') == 1:
	    # source + footer
	    first = rpc_src_lines.index("%%")
	    src = '\n'.join(rpc_src_lines[:first])
	    footer = '\n'.join(rpc_src_lines[first + 1:])
    else: 
        src = '\n'.join(rpc_src_lines) 
      
    rpc_source = parse("rpc_source", src)
    with open(os.path.splitext(rpc_fpath)[0] + ".h", "w") as f:
        f = SourceFile(f)
        f.writeln("// generated from '%s'" % os.path.split(rpc_fpath)[1])
        f.writeln()
        f.writeln("#pragma once")
        f.writeln()
        f.writeln('#include "rpc/server.h"')
        f.writeln('#include "rpc/client.h"')
        f.writeln()
        f.writeln("#include <errno.h>")
        f.writeln()
        f.write(header)
        emit_rpc_source(rpc_source, f)
        f.write(footer)

if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.stdout.write("usage: %s <rpc-source-file>\n" % sys.argv[0])
        exit(1)
    rpcgen(sys.argv[1])
