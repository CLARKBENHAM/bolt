import os 
os.chdir(os.path.dirname(__file__))

MITHRAL_BUILD="./bazel-bin/mithral_wrapped.so"
if not os.path.isfile(MITHRAL_BUILD):
  raise Exception(f"Expected file {MITHRAL_BUILD} Was `bazel build :mithral_wrapped` run?")

try:
	os.symlink(MITHRAL_BUILD, "mithral_wrapped.so")
except FileExistsError:
  pass

##works
from . import mithral_wrapped
