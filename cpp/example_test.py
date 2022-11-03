import os
import sys

#print('about to import', file=sys.stderr)
#print('python is', sys.version_info)
#print('pid is', os.getpid())

#import mithral_wrapped
from cpp import mithral_wrapped

#print('imported, about to call', file=sys.stderr)

#Test pybind11 wrapped correctly
result =mithral_wrapped.add(2, 3)
assert result == 5
#print(dir(mithral_wrapped), mithral_wrapped.add(mithral_wrapped.sub(2, 3), 3))

#Run imported test cases
my_str = "Caltech3x3"
my_str_as_bytes = str.encode(my_str)
name=my_str#_as_bytes #my_str_as_bytes
##name=bytearray("Caltech3x3", 'utf-8')
#print(type(my_str_as_bytes)) # ensure it is byte representation
#my_decoded_str = my_str_as_bytes.decode()
#print(type(my_decoded_str)) 
#print('std::string', mithral_wrapped.utf8_test(name))
#print('charptr', mithral_wrapped.utf8_charptr(name))

C=mithral_wrapped.clark(name)
print("getNameString", C.getNameString())
print("getName", C.getName())
print("Clark", C.c)
print("Clark worked\n\n")


N,D,M = (224 - 3 + 1) * (224 - 3 + 1), 3 * (3 * 3), 2
T = mithral_wrapped.Test(N,D,M, name)
print("TESTING: ", T.name)
mithral_wrapped.printNameTest(T) #get this to work and pybinding work
print(T.N, T.D, T.M)
print(T, name)
print("\n\n")

print("MatmulTaskShape")
N,D,M = (224 - 3 + 1) * (224 - 3 + 1), 3 * (3 * 3), 2
matmul= mithral_wrapped.MatmulTaskShape(N,D,M, name)
print("TESTING: ", matmul.name)
mithral_wrapped.printNameMatmul(matmul) #get this to work and pybinding work
print(matmul.N, matmul.D, matmul.M)
print(matmul, name)
print("\n\n")
ncodebooks= [64] #[2, 4, 8, 16, 32, 64]
lutconsts= [-1] #[-1, 1, 2, 4]
#mithral_wrapped._profile_mithral(matmul, ncodebooks, lutconsts)

N,D,M = (224 - 3 + 1) * (224 - 3 + 1), 3 * (3 * 3), 2
kCaltechTaskShape0=mithral_wrapped.MatmulTaskShape(N,D,M, name)
#kCaltechTaskShape0.name=name #if I assigned to it then code sometimes gets corrupted
print(type(kCaltechTaskShape0))
print('N: ', kCaltechTaskShape0.N) 
print('D: ', kCaltechTaskShape0.D) 
print('M: ', kCaltechTaskShape0.M) 
print('name: ', kCaltechTaskShape0.name) #errors
print(kCaltechTaskShape0, name)
mithral_wrapped.printNameMatmul(kCaltechTaskShape0) #get this to work and pybinding work

mithral_wrapped._profile_mithral(kCaltechTaskShape0, ncodebooks, lutconsts)
##a bit faster
mithral_wrapped._profile_mithral_int8(kCaltechTaskShape0, ncodebooks, lutconsts)

print('done!', file=sys.stderr)