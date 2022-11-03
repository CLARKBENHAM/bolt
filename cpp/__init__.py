import os 
try:
	os.symlink("./bazel-bin/mithral_wrapped.so", "mithral_wrapped.so")
except FileExistsError:
  pass

##works
from . import mithral_wrapped
