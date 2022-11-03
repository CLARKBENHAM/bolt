import re
import os
import sys
import io
from contextlib import contextmanager
import ctypes
import tempfile
from collections import namedtuple

from cpp import mithral_wrapped

#Test pybind11 wrapped correctly
result =mithral_wrapped.add(2, 3)
assert result == 5
#print(dir(mithral_wrapped), mithral_wrapped.add(mithral_wrapped.sub(2, 3), 3))


#So C++ code redirected to python
libc = ctypes.CDLL(None)
c_stdout = ctypes.c_void_p.in_dll(libc, 'stdout')
@contextmanager
def stdout_redirector(stream):
    # The original fd stdout points to. Usually 1 on POSIX systems.
    original_stdout_fd = sys.stdout.fileno()

    def _redirect_stdout(to_fd):
        """Redirect stdout to the given file descriptor."""
        # Flush the C-level buffer stdout
        libc.fflush(c_stdout)
        # Flush and close sys.stdout - also closes the file descriptor (fd)
        sys.stdout.close()
        # Make original_stdout_fd point to the same file as to_fd
        os.dup2(to_fd, original_stdout_fd)
        # Create a new sys.stdout that points to the redirected fd
        sys.stdout = io.TextIOWrapper(os.fdopen(original_stdout_fd, 'wb'))

    # Save a copy of the original stdout fd in saved_stdout_fd
    saved_stdout_fd = os.dup(original_stdout_fd)
    try:
        # Create a temporary file and redirect stdout to it
        tfile = tempfile.TemporaryFile(mode='rb')
        _redirect_stdout(tfile.fileno())
        # Yield to caller, then redirect stdout back to the saved fd
        yield
        _redirect_stdout(saved_stdout_fd)
        # Copy contents of temporary file to the given stream
        tfile.flush()
        tfile.seek(0, io.SEEK_SET)
        stream.write(tfile.read())
    finally:
        tfile.close()
        os.close(saved_stdout_fd)

#Run imported test cases
name = "Caltech3x3"
N,D,M = (224 - 3 + 1) * (224 - 3 + 1), 3 * (3 * 3), 2
kCaltechTaskShape0=mithral_wrapped.MatmulTaskShape(N,D,M, name)
#kCaltechTaskShape0.name=name #if I assigned to it then code gets corrupted
print(type(kCaltechTaskShape0))
print('N, D,M: ', kCaltechTaskShape0.N, kCaltechTaskShape0.D, kCaltechTaskShape0.M) 
print('name: ', kCaltechTaskShape0.name) #errors
print(kCaltechTaskShape0, name)
mithral_wrapped.printNameMatmul(kCaltechTaskShape0) #get this to work and pybinding work

cpp_timing_results=namedtuple('cpp_timing_results', "dataset_name numeric_type problem N D M C lut_work_coef n_reps n_trials trial_best_mult_ops_per_sec")
def extract_catch_tests(data):
  if data[:3] in ('MSG','---', '   ') or len(data) < 5:
    return False
  try:
    dataset_name, numeric_type, problem, _, N,D,M,C,lut_work_coef,n_reps_trials, *results =data.split(",")
    N,D,M,C, lut_work_coef = map(float, [N,D,M,C, lut_work_coef])
    n_reps, n_trials = map(int, re.findall('\d+', n_reps_trials))
    trial_best_mult_ops_per_sec = [float(re.search('[\d\.]+e\+\d+', i).group(0)) for i in results[1::2]]
    return cpp_timing_results(dataset_name, numeric_type, problem.strip(), N,D,M,C,lut_work_coef, n_reps, n_trials, trial_best_mult_ops_per_sec)
  except Exception as e:
    print("Ignoring", e, type(e).__name__, data)
    return False

buf=io.BytesIO()
with stdout_redirector(buf):
  ncodebooks= [64] #[2, 4, 8, 16, 32, 64]
  lutconsts= [-1] #[-1, 1, 2, 4]
  mithral_wrapped._profile_mithral(kCaltechTaskShape0, ncodebooks, lutconsts)
s=buf.getvalue().decode()
data = list(filter( lambda i: i, map(extract_catch_tests, s.split("\n"))))
print('data', data)
 
##a bit faster
mithral_wrapped._profile_mithral_int8(kCaltechTaskShape0, ncodebooks, lutconsts)

print('done!', file=sys.stderr)