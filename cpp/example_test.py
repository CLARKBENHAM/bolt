import os
import sys

print('about to import', file=sys.stderr)
print('python is', sys.version_info)
print('pid is', os.getpid())

#import mithral_wrapped
from cpp import mithral_wrapped

print('imported, about to call', file=sys.stderr)

result =mithral_wrapped.add(2, 3)
print(result)
assert result == 5
print(dir(mithral_wrapped), mithral_wrapped.add(mithral_wrapped.sub(2, 3), 3))

print('done!', file=sys.stderr)