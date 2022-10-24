import os  #it only works if run in cpp/ first to create the symlink then it exists elsewhere
try:
	os.symlink("./bazel-bin/mithral_wrapped.so", "mithral_wrapped.so")
except Exception as e:
  #print('ignoring', e)
  pass

##works
from . import mithral_wrapped