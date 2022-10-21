import os
import sys

print('about to import', file=sys.stderr)
print('python is', sys.version_info)
print('pid is', os.getpid())

import pybind11_wrapper 

print('imported, about to call', file=sys.stderr)

result = pybind11_wrapper.add(2, 3)
print(result)
assert result == 5

print('done!', file=sys.stderr)


#import example
##from example import add,sub
#assert(9, example.add(9,example.sub(9,0)))