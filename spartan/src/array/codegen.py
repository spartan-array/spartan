import sys

types = ['npy_bool',
         'npy_int', 
         'npy_uint', 
         'npy_longlong', 
         'npy_ulonglong',
         'npy_float',
         'npy_double']

type_ltrs = ['NPY_BOOLLTR',
             'NPY_INTLTR', 
             'NPY_UINTLTR', 
             'NPY_LONGLONGLTR', 
             'NPY_ULONGLONGLTR',
             'NPY_FLOATLTR',
             'NPY_DOUBLELTR']

type_names = ['BOOL',
              'INT',
              'UINT',
              'LONGLONG',
              'ULONGLONG',
              'FLOAT',
              'DOUBLE']

def replace(program, replace_begin, replace_end):
  output = ''
  for i in range(len(type_names)):
    begin = replace_begin
    end = replace_end
    while True:
      index = program.find('_RP_', begin, end);
      if index == -1:
        break
      output += program[begin:index]
      if program[index:index + len('_RP_TYPE_')] == '_RP_TYPE_':
          output += types[i]
          begin = index + len('_RP_TYPE_')
      elif program[index:index + len('_RP_TYPELTR_')] == '_RP_TYPELTR_':
          output += type_ltrs[i]
          begin = index + len('_RP_TYPELTR_')
      elif program[index:index + len('_RP_NAME_')] == '_RP_NAME_':
          output += type_names[i]
          begin = index + len('_RP_NAME_')
      else:
        assert False, (program[begin:end])
    output += program[begin:end]
  return output

def main():
  with open(sys.argv[1]) as rfp:
    with open(sys.argv[1][:-4], 'w') as wfp:
      # Read all lines at once. It can simply the process.
      program = rfp.read() 
      new_program = ''
      begin = end = 0
      replace_begin = replace_end = 0
      while True:
        replace_begin = program.find('/*_RP_BEGIN_*/', begin)
        if replace_begin == -1:
          break
        end = replace_begin - 1
        replace_begin += len('/*_RP_BEGIN_*/')
        replace_end = program.find('/*_RP_END_*/', replace_begin)
        assert(replace_end != -1)

        new_program += program[begin:end + 1]
        new_program += replace(program, replace_begin, replace_end)
        begin = replace_end + len('/*_RP_END_*/')

      new_program += program[begin:]
      wfp.write(new_program)
      
if __name__ == '__main__':
  main()
