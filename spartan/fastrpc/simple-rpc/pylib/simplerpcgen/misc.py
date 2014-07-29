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
