import os 
os.chdir(os.path.dirname(__file__))
try:
	os.symlink("./bazel-bin/mithral_wrapped.so", "mithral_wrapped.so")
except FileExistsError:
  pass

##works
from . import mithral_wrapped
