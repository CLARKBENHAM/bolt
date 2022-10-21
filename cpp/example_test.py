import os
import sys

print('about to import', file=sys.stderr)
print('python is', sys.version_info)
print('pid is', os.getpid())

#import my_pb_mod
import pybind11_wrapper as my_pb_mod

print('imported, about to call', file=sys.stderr)

result = my_pb_mod.add(2, 3)
print(result)
assert result == 5
print(dir(my_pb_mod), my_pb_mod.add(my_pb_mod.sub(2, 3), 3))

print('done!', file=sys.stderr)