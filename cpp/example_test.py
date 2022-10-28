import os
import sys

print('about to import', file=sys.stderr)
print('python is', sys.version_info)
print('pid is', os.getpid())

#import mithral_wrapped
from cpp import mithral_wrapped

print('imported, about to call', file=sys.stderr)

#Test pybind11 wrapped correctly
result =mithral_wrapped.add(2, 3)
print(result)
assert result == 5
print(dir(mithral_wrapped), mithral_wrapped.add(mithral_wrapped.sub(2, 3), 3))

#Run imported test cases
my_str = "Caltech3x3"
my_str_as_bytes = str.encode(my_str)
print(type(my_str_as_bytes)) # ensure it is byte representation
my_decoded_str = my_str_as_bytes.decode()
print(type(my_decoded_str)) 
name=my_str_as_bytes
#name=bytearray("Caltech3x3", 'utf-8')

N,D,M = (224 - 3 + 1) * (224 - 3 + 1), 3 * (3 * 3), 2
kCaltechTaskShape0=mithral_wrapped.MatmulTaskShape(N,D,M, name)
ncodebooks=[2, 4, 8, 16, 32, 64]
lutconsts=[-1, 1, 2, 4]
print(type(kCaltechTaskShape0))
#print(kCaltechTaskShape0.name) #errors
#print(kCaltechTaskShape0)
out=mithral_wrapped._profile_mithral(kCaltechTaskShape0, ncodebooks, lutconsts)

print('done!', file=sys.stderr)