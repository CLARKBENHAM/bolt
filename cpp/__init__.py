import os  
try:
	os.symlink("./bazel-bin/mithral_wrapped.so", "mithral_wrapped.so")
except Exception as e:
  #print('ignoring', e)
  pass

##works
from . import mithral_wrapped