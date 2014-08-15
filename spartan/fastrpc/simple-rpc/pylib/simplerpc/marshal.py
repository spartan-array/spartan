import collections
from simplerpc import _pyrpc

class Marshal(object):

    # class variable
    structs = {} # typename -> (ctor, [(field_name, field_type)])

    @staticmethod
    def reg_type(type, fields):
        ctor = collections.namedtuple(type, [field[0] for field in fields])
        Marshal.structs[type] = ctor, fields
        return ctor

    def __init__(self, id=None, should_release=True):
        if not id:
            self.id = _pyrpc.init_marshal()
        else:
            self.id = id
        self.should_release = should_release

    def __del__(self):
        if self.should_release:
            _pyrpc.fini_marshal(self.id)

    def __len__(self):
        return _pyrpc.marshal_size(self.id)

    def write_i8(self, i8):
        _pyrpc.marshal_write_i8(self.id, i8)

    def read_i8(self):
        return _pyrpc.marshal_read_i8(self.id)

    def write_i16(self, i16):
        _pyrpc.marshal_write_i16(self.id, i16)

    def read_i16(self):
        return _pyrpc.marshal_read_i16(self.id)

    def write_i32(self, i32):
        _pyrpc.marshal_write_i32(self.id, i32)

    def read_i32(self):
        return _pyrpc.marshal_read_i32(self.id)

    def write_i64(self, i64):
        _pyrpc.marshal_write_i64(self.id, i64)

    def read_i64(self):
        return _pyrpc.marshal_read_i64(self.id)

    def write_v32(self, v32):
        _pyrpc.marshal_write_v32(self.id, v32)

    def read_v32(self):
        return _pyrpc.marshal_read_v32(self.id)

    def write_v64(self, v64):
        _pyrpc.marshal_write_v64(self.id, v64)

    def read_v64(self):
        return _pyrpc.marshal_read_v64(self.id)

    def write_double(self, dbl):
        _pyrpc.marshal_write_double(self.id, dbl)

    def read_double(self):
        return _pyrpc.marshal_read_double(self.id)

    def write_str(self, s):
        _pyrpc.marshal_write_str(self.id, s)

    def read_str(self):
        return _pyrpc.marshal_read_str(self.id)

    @staticmethod
    def template_split(type_str):
        splt = []
        level = 0
        sp = ""
        for c in type_str:
            if c == "<":
                level += 1
            elif c == ">":
                level -= 1
            if c == "," and level == 0:
                splt += sp.strip(),
                sp = ""
            else:
                sp += c
        if sp.strip() != "":
            splt += sp.strip(),
        return splt

    # write list/tuple/dict/set
    def write_obj(self, o, obj_t):
        if obj_t in ["rpc::i8", "i8"]:
            return self.write_i8(o)
        elif obj_t in ["rpc::i16", "i16"]:
            return self.write_i16(o)
        elif obj_t in ["rpc::i32", "i32"]:
            return self.write_i32(o)
        elif obj_t in ["rpc::i64", "i64"]:
            return self.write_i64(o)
        elif obj_t in ["rpc::v32", "v32"]:
            return self.write_v32(o)
        elif obj_t in ["rpc::v64", "v64"]:
            return self.write_v64(o)
        elif obj_t == "double":
            return self.write_double(o)
        elif obj_t in ["std::string", "string"]:
            return self.write_str(o)
        elif obj_t.startswith("std::pair<"):
            first_t, second_t = Marshal.template_split(obj_t[obj_t.index("<") + 1:-1])
            self.write_obj(o[0], first_t)
            self.write_obj(o[1], second_t)
        elif obj_t.startswith("std::map<") or obj_t.startswith("std::unordered_map<"):
            key_t, val_t = Marshal.template_split(obj_t[obj_t.index("<") + 1:-1])
            self.write_v64(len(o))
            for k in o:
                self.write_obj(k, key_t)
                self.write_obj(o[k], val_t)
        elif obj_t.startswith("std::vector<") or obj_t.startswith("std::list<") or obj_t.startswith("std::set<") or obj_t.startswith("std::unordered_set<"):
            val_t = obj_t[obj_t.index("<") + 1:-1]
            self.write_v64(len(o))
            for v in o:
                self.write_obj(v, val_t)
        else:
            ty = Marshal.structs[obj_t][1]
            for field in ty:
                self.write_obj(getattr(o, field[0]), field[1])

    # read list/dict/set, tuple will be read as list, pair will be read as tuple
    def read_obj(self, obj_t):
        if obj_t in ["rpc::i8", "i8"]:
            return self.read_i8()
        elif obj_t in ["rpc::i16", "i16"]:
            return self.read_i16()
        elif obj_t in ["rpc::i32", "i32"]:
            return self.read_i32()
        elif obj_t in ["rpc::i64", "i64"]:
            return self.read_i64()
        elif obj_t in ["rpc::v32", "v32"]:
            return self.read_v32()
        elif obj_t in ["rpc::v64", "v64"]:
            return self.read_v64()
        elif obj_t == "double":
            return self.read_double()
        elif obj_t in ["std::string", "string"]:
            return self.read_str()
        elif obj_t.startswith("std::pair<"):
            first_t, second_t = Marshal.template_split(obj_t[obj_t.index("<") + 1:-1])
            first = self.read_obj(first_t)
            second = self.read_obj(second_t)
            return (first, second)
        elif obj_t.startswith("std::map<") or obj_t.startswith("std::unordered_map<"):
            key_t, val_t = Marshal.template_split(obj_t[obj_t.index("<") + 1:-1])
            d = {}
            n = self.read_v64()
            for i in range(n):
                k = self.read_obj(key_t)
                v = self.read_obj(val_t)
                d[k] = v
            return d
        elif obj_t.startswith("std::vector<") or obj_t.startswith("std::list<"):
            val_t = obj_t[obj_t.index("<") + 1:-1]
            lst = []
            n = self.read_v64()
            for i in range(n):
                o = self.read_obj(val_t)
                lst += o,
            return lst
        elif obj_t.startswith("std::set<") or obj_t.startswith("std::unordered_set<"):
            val_t = obj_t[obj_t.index("<") + 1:-1]
            st = set()
            n = self.read_v64()
            for i in range(n):
                o = self.read_obj(val_t)
                st.add(o)
            return st
        else:
            ty = Marshal.structs[obj_t][1]
            field_values = []
            for field in ty:
                field_values += self.read_obj(field[1]),
            return Marshal.structs[obj_t][0](*field_values)
