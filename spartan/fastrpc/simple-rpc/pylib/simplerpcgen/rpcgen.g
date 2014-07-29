import sys
import os
import random
import re
sys.path += os.path.abspath(os.path.join(os.path.split(__file__)[0], "../../pylib")),
from simplerpcgen.lang_cpp import emit_rpc_source_cpp
from simplerpcgen.lang_python import emit_rpc_source_python

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
    if t in ["pair", "string", "map", "list", "set", "vector", "unordered_map", "unordered_set"]:
        t = "std::" + t
    return t

def forbid_reserved_names(name):
    if re.match("__([^_]+.*[^_]+|[^_])__$", name):
        raise Exception("bad name '%s', __NAME__ format names are reserved" % name)

def check_rpc_func(attrs, output):
    if ("defer" in attrs) and ("udp" in attrs):
        raise Exception("udp functions does not have return value, cannot provide deferred return values")
    if ("raw" in attrs) and ("defer" in attrs):
        raise Exception("cannot generate deferred return code stub for raw RPC handler")
    if ("raw" in attrs) and ("fast" in attrs):
        raise Exception("cannot generate fast RPC code stub for raw RPC handler")
    if ("fast" in attrs) and ("defer" in attrs):
        raise Exception("cannot mark an RPC as both doing fast return and deferred return")
    if ("udp" in attrs) and len(output) > 0:
        raise Exception("udp RPC handler cannot have return value")

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
        "i8" {{ return "rpc::i8" }}
        | "i16" {{ return "rpc::i16" }}
        | "i32" {{ return "rpc::i32" }}
        | "i64" {{ return "rpc::i64" }}
        | "v32" {{ return "rpc::v32" }}
        | "v64" {{ return "rpc::v64" }}
        | full_symbol {{ t = std_rename(full_symbol) }}
            ["<" type {{ t += "<" + type }} ("," type {{ t += ", " + type }})* ">" {{ t += ">" }}] {{ return t }}
        | ("bool" | "int" | "unsigned" | "long" ) {{ error("please use i8, i16, i32, i64, v32 or v64 instead", _context) }}

    rule full_symbol: {{ s = "" }}
        ["::" {{ s += "::" }}] SYMBOL {{ s += SYMBOL }} ("::" SYMBOL {{ s += "::" + SYMBOL }})*
            {{ return s }}

    rule service_decl: {{ abstract = False }}
        ["abstract" {{ abstract = True }}] "service" SYMBOL "{" service_functions "}"
            {{ return pack(name=SYMBOL, abstract=abstract, functions=service_functions) }}

    rule service_functions: {{ functions = [] }}
        (service_function {{ functions += service_function, }})*
            {{ return functions }}

    rule service_function: {{ abstract = False; input = []; output = [] }}
        func_attrs SYMBOL {{ forbid_reserved_names(SYMBOL) }}
        "\(" (func_arg_list {{ input = func_arg_list }}) ["\|" (func_arg_list {{ output = func_arg_list }})] "\)"
        ["=" "0" {{ abstract = True }}]
            {{ check_rpc_func(attrs=func_attrs, output=output) }}
            {{ return pack(name=SYMBOL, attrs=func_attrs, abstract=abstract, input=input, output=output) }}

    rule func_attrs: {{ attrs = set() }}
        (func_attr {{ attrs.add(func_attr,) }})*
            {{ return attrs }}

    rule func_attr:
        "fast" {{ return "fast" }}
        | "raw" {{ return "raw" }}
        | "defer" {{ return "defer" }}
        | "udp" {{ return "udp" }}

    rule func_arg_list: {{ args = [] }}
        (| func_arg {{ args = [func_arg] }} ("," func_arg {{ args += func_arg, }})*)
            {{ return args }}

    rule func_arg: {{ name = None }}
        type [SYMBOL {{ name = SYMBOL; forbid_reserved_names(name) }}]
            {{ return pack(name=name, type=type) }}

%%

def generate_rpc_table(rpc_source):
    rpc_table = {}
    for service in rpc_source.services:
        for func in service.functions:
            rpc_code = random.randint(0x10000000, 0x70000000)
            rpc_table["%s.%s" % (service.name, func.name)] = rpc_code
    return rpc_table

def rpcgen(rpc_fpath, languages):
    with open(rpc_fpath) as f:
        rpc_src = f.read()

    rpc_src_lines = rpc_src.split("\n")

    cpp_header = cpp_footer = src = ''

    if rpc_src_lines.count('%%') == 2:
        # cpp_header + source + cpp_footer
        first = rpc_src_lines.index("%%")
        next = rpc_src_lines.index("%%", first + 1)
        cpp_header =  '\n'.join(rpc_src_lines[:first])
        src = '\n'.join(rpc_src_lines[first + 1:next])
        cpp_footer = '\n'.join(rpc_src_lines[next + 1:])
    elif rpc_src_lines.count('%%') == 1:
        # source + cpp_footer
        first = rpc_src_lines.index("%%")
        src = '\n'.join(rpc_src_lines[:first])
        cpp_footer = '\n'.join(rpc_src_lines[first + 1:])
    else:
        src = '\n'.join(rpc_src_lines)

    rpc_source = parse("rpc_source", src)
    rpc_table = generate_rpc_table(rpc_source) # service.func = rpc_code

    if "cpp" in languages:
        fpath = os.path.splitext(rpc_fpath)[0] + ".h"
        emit_rpc_source_cpp(rpc_source, rpc_table, fpath, cpp_header, cpp_footer)

    if "python" in languages:
        fpath = os.path.splitext(rpc_fpath)[0] + ".py"
        emit_rpc_source_python(rpc_source, rpc_table, fpath)
