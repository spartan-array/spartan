import sys

def GenDecl(N):
    decl   = [ "template<typename Res" ]
    for i in range(1,N+2):
        decl.append(", typename Args%d=void" % i)
    decl.append(">\n")
    decl.append("class Callback;\n\n")
    return ''.join(decl)


def GenIthBase(i):
    base   = [ "template<%s>",                             # decls
               "class Callback<%s> {",                     # types
               "public:",
               "  virtual ~Callback() {}",
               "",
               "  virtual Res run(%s) = 0;",               # params
               "  Res operator()(%s) { return this->run(%s); }",  # params_decl, params_ins
               "  virtual bool once() const { return false; }", # default once() to compatible with Runnable()
               "};",
               "",
               ""]

    decls  = [ "typename Res" ]
    types  = [ "Res" ]
    params = [ ]
    params_decl = [ ]
    params_ins = [ ]
    for arg in range(1,i+1):
        arg_str = "Arg%d" % arg
        decls.append("typename %s" % arg_str)
        types.append(arg_str)
        params.append(arg_str)
        params_decl.append(arg_str + " " + arg_str.lower())
        params_ins.append(arg_str.lower())

    expanded = (','.join(decls), ','.join(types), ','.join(params),
                ', '.join(params_decl), ', '.join(params_ins))
    return '\n'.join(base) % expanded


def GenIthInstance(i):
    base   = [ "template<typename Target, typename Res%s>", #1 ,typename types
               "class %s :",                                #2 class name
               "  public Callback<Res%s> {",                #3 , not-bound list
               "",
               "public:",
               "  typedef Res(Target::*TargetFunc)(%s);",   #4 types list
               "",
               "  %s(TargetFunc target_func,",              #5 class name
               "    Target* obj%s)",                        #6 , bound param list
               "    : target_func_(target_func),",
               "      obj_(obj)%s { }",                     #7 init bound
               "",
               "  virtual ~%s() {}",                        #8 class name
               "",
               "  virtual Res run(%s) {",                   #9 not-bound decl
               "    %s",                                    #10 () body
               "  }",
               "",
               "  virtual bool once() const {",
               "    return %s;",                            #11 true | false
               "  }",
               "",
               "private:",
               "  TargetFunc target_func_; // not owned here",
               "  Target* obj_;            // not owned here",
               "%s",                                        #12 bound defn
               "};",
               "" ]

    spec   = [ "template<typename Target%s>",               #1 ,typename types
               "class %s<Target, void%s> :",                #2 class name
                                                            #2a, type list
               "  public Callback<void%s> {",               #3 , not-bound list
               "",
               "public:",
               "  typedef void(Target::*TargetFunc)(%s);",  #4 types list
               "",
               "  %s(TargetFunc target_func,",              #5 class name
               "    Target* obj%s)",                        #6 , bound param list
               "    : target_func_(target_func),",
               "      obj_(obj)%s { }",                     #7 init bound
               "",
               "  virtual ~%s() {}",                        #8 class name
               "",
               "  virtual void run(%s) {",                  #9 not-bound decl
               "    %s",                                    #10 () body
               "  }",
               "",
               "  virtual bool once() const {",
               "    return %s;",                            #11 true | false
               "  }",
               "",
               "private:",
               "  TargetFunc target_func_; // not owned here",
               "  Target* obj_;            // not owned here",
               "%s",                                        #12 bound defn
               "};",
               "" ]

    factory= [ "template<typename Target, typename Res%s>", #1 , typename types
               "%s<Target, Res%s>*",                        #2 class name
                                                            #3 , types list
               "make%s(",                                   #4 factory name
               "    Res (Target::*f)(%s),",                 #5 types list
               "    Target* obj%s) {",                      #6 , bound param list
               "  return new %s<Target,Res%s>(",            #7 class name
                                                            #8 , types list
               "    f,",
               "    obj%s);",                               #9 , bind names
               "}",
               "" ]

    # Possible bodies for run()
    one    = [     "Res ret = ((*obj_).*target_func_)(%s);", # bound + unbound
               "    delete this;",
               "    return ret;" ]

    void   = [     "((*obj_).*target_func_)(%s);",           # bound + unbound
               "    delete this;" ]

    many   =       "return ((*obj_).*target_func_)(%s);"     # ditto

    # Make the transient and the permanent variants
    bodies = [ "Callable", "CallableOnce" ]

    res = []
    bound = 0;
    while True:
        for once in [True, False]:
            nbound = i-bound;
            rgn = range(bound);
            rgnn = range(nbound);

            bound_tp = ["Bind%d" % x for x in range(1, bound+1)]
            bound_name = ["bind%d" % x for x in range(1, bound+1)]
            bound_name_u = ["bind%d_" % x for x in range(1, bound+1)]
            nbound_tp = ["Arg%d" % x for x in range(1, i-bound+1)]
            nbound_name = ["arg%d" % x for x in range(1, i-bound+1)]

            skip4 = ",\n    "
            skip6 = ",\n      "

            class_name = "%s_%d_%d" % (bodies[once], i, bound)
            typenames = [", typename %s" % x for x in bound_tp + nbound_tp]
            types = ["%s" % x for x in bound_tp + nbound_tp]
            ctypes = [", %s" % x for x in bound_tp + nbound_tp]
            cbnames = [skip4+"%s" % x for x in bound_name]
            names = ["%s" % x for x in bound_name_u + nbound_name]
            parms = [skip4+"%s %s" % (bound_tp[x], bound_name[x]) for x in rgn]
            init = [skip6+"%s_(%s)" % (bound_name[x], bound_name[x]) for x in rgn]
            decl = ["  %s %s_;" % (bound_tp[x], bound_name[x]) for x in rgn]
            ntypes = [", %s" % x for x in nbound_tp]
            ndecl = ["%s %s" % (nbound_tp[x], nbound_name[x]) for x in rgnn]

            if once:
                body = '\n'.join(one) % ', '.join(names)
                body_spec = '\n'.join(void) % ', '.join(names)
                once_str = "true"
            else:
                body = many % ', '.join(names)
                once_str = "false"

            expanded = (''.join(typenames),      #1
                        class_name,              #2
                        ''.join(ntypes),         #3
                        ', '.join(types),        #4
                        class_name,              #5
                        ''.join(parms),          #6
                        ''.join(init),           #7
                        class_name,              #8
                        ', '.join(ndecl),        #9
                        body,                    #10
                        once_str,                #11
                        '\n'.join(decl))         #12
            res.append('\n'.join(base) % expanded)

            if once:
                expanded = (''.join(typenames),  #1
                            class_name,          #2
                            ''.join(ctypes),     #2a
                            ''.join(ntypes),     #3
                            ', '.join(types),    #4
                            class_name,          #5
                            ''.join(parms),      #6
                            ''.join(init),       #7
                            class_name,          #8
                            ', '.join(ndecl),    #9
                            body_spec,           #10
                            once_str,            #11
                            '\n'.join(decl))     #12
                res.append('\n'.join(spec) % expanded)

            expanded = ( ''.join(typenames),     #1
                         class_name,             #2
                         ''.join(ctypes),        #3
                         bodies[once],           #4
                         ', '.join(types),       #5
                         ''.join(parms),         #6
                         class_name,             #7
                         ''.join(ctypes),        #8
                         ''.join(cbnames))       #9
            res.append('\n'.join(factory) % expanded)

        bound += 1
        if (bound > i):
            break;

    return '\n'.join(res)


def GenFile(N):
    # Warning not to overwrite generated file.
    warn   = [ "// This file was generated by 'callback.py'.",
               "// Do not hand-edit anything here.",
               "" ]

    # Include guard for the generated file and open namespace.
    header = [ "#ifndef CALLBACK_INSTANCES_HEADER",
               "#define CALLBACK_INSTANCES_HEADER",
               "",
               "namespace rpc {",
               "",
               ""]

    # End namespace and include guard.
    footer = [ ""
               "}  // namespace rpc",
               "",
               "#endif  // CALLBACK_INSTANCES_HEADER" ]

    out = open("callback_instances.h", 'w')
    out.truncate()
    out.write('\n'.join(warn))
    out.write('\n'.join(header))
    out.write(GenDecl(N))
    for i in range(N+1):
        out.write(GenIthBase(i))
    for i in range(N+1):
        out.write(GenIthInstance(i))
    out.write('\n'.join(footer))
    out.write('\n')


def main(argv):
    """Usage:\n$ python callback.py [N]"""

    count = len(argv)
    if count == 1:
        N = 3  # instantiate classes for max 3 arguments

    elif count == 2:
        N = int(argv[-1])
        if N < 0 or N > 10:
            print "Invalid argument %r" % argv[-1]
            print main.__doc__
            sys.exit(1)

    else:
        print main.__doc__
        sys.exit(1)

    GenFile(N)


if __name__ == '__main__':
    main(sys.argv)
