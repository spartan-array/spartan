from pytable import util
from pytable.array import expr, compile_expr

y = expr.LazyVal(0)
z = y * y + y

c = compile_expr.compile_op(z)
util.log('RESULT:')
util.log(c.to_str(0))