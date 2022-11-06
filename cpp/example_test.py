import re
import os
import sys
import io
from contextlib import contextmanager
import ctypes
import tempfile
from collections import namedtuple
from timeit import default_timer as timer
import numpy as np

from cpp import mithral_wrapped

# Test pybind11 wrapped correctly
result =mithral_wrapped.add(2, 3)
assert result == 5


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

cpp_timing_results=namedtuple('cpp_timing_results', "dataset_name numeric_type problem N D M C lut_work_coef n_reps n_trials trial_best_mult_ops_per_sec")


def _extract_cpp_speed(data):
  """parses Catch2 test printout.
  Likely to break
  """
  if data[:3] in ('MSG','---', '   ') or len(data) < 5:
    return False
  try:
    dataset_name, \
    numeric_type, \
    problem, \
    _, N,D,M,C,lut_work_coef, \
    n_reps_trials,\
    *results =data.split(",")
    N,D,M,C, lut_work_coef = map(float, [N,D,M,C, lut_work_coef])
    n_reps, n_trials = map(int, re.findall('\d+', n_reps_trials))
    trial_best_mult_ops_per_sec = [float(re.search('[\d\.]+e\+\d+', i).group(0))
                                   for i in results[1::2]]
    return cpp_timing_results(dataset_name.strip(),
                              numeric_type.strip(),
                              problem.strip(),
                              N,D,M,C,lut_work_coef,
                              n_reps,
                              n_trials,
                              trial_best_mult_ops_per_sec)
  except Exception as e:
    print("Ignoring", e, type(e).__name__, data)
    return False

def  run_cpp_timing(catch_tests_fn, extract_fn=_extract_cpp_speed):
  """catch_tests_fn starts a process that prints timing test results to stdout.
  extract_fn parses a row of stdout
  """
  buf=io.BytesIO()
  with stdout_redirector(buf):
   catch_tests_fn()
  s=buf.getvalue().decode()
  data = list(filter( lambda i: i, map(extract_fn, s.split("\n"))))
  return data


name = 'cifar100' #"Caltech3x3"
N,D,M = (224 - 3 + 1) * (224 - 3 + 1), 3 * (3 * 3), 2
kCaltechTaskShape0=mithral_wrapped.MatmulTaskShape(N,D,M, name)

def test_pybinding(kCaltechTaskShape0):
  #kCaltechTaskShape0.name=name #if I assigned to it then name gets corrupted/errorrs
  print(type(kCaltechTaskShape0))
  print('N, D,M: ', kCaltechTaskShape0.N, kCaltechTaskShape0.D, kCaltechTaskShape0.M)
  print('name: ', kCaltechTaskShape0.name) 
  print(kCaltechTaskShape0, name)
  mithral_wrapped.printNameMatmul(kCaltechTaskShape0) 

ncodebooks= [64] #[2, 4, 8, 16, 32, 64]
lutconsts= [-1] #[-1, 1, 2, 4]
float32_profile = lambda : mithral_wrapped._profile_mithral(kCaltechTaskShape0, ncodebooks, lutconsts)
int8_profile = lambda :mithral_wrapped._profile_mithral_int8(kCaltechTaskShape0, ncodebooks, lutconsts)

#float32_data=run_cpp_timing(float32_profile)
#int8_data = run_cpp_timing(int8_profile)
float32_data=[cpp_timing_results(dataset_name='Caltech3x3', numeric_type=' f32', problem='amm mithral nolut', N=49284.0, D=27.0, M=2.0, C=64.0, lut_work_coef=-1.0, n_reps=5, n_trials=10, trial_best_mult_ops_per_sec=[22409000.0, 23713000.0, 22558000.0, 22615000.0, 22845000.0]), cpp_timing_results(dataset_name='Caltech3x3', numeric_type=' f32', problem='amm mithral denselut', N=49284.0, D=27.0, M=2.0, C=64.0, lut_work_coef=-1.0, n_reps=5, n_trials=10, trial_best_mult_ops_per_sec=[22404000.0, 22476000.0, 22273000.0, 22029000.0, 22641000.0]), cpp_timing_results(dataset_name='Caltech3x3', numeric_type=' f32', problem='mithral lut dense', N=49284.0, D=27.0, M=2.0, C=64.0, lut_work_coef=-1.0, n_reps=5, n_trials=10, trial_best_mult_ops_per_sec=[16437000000.0, 16437000000.0, 16437000000.0, 16437000000.0, 16437000000.0]), cpp_timing_results(dataset_name='Caltech3x3', numeric_type=' f32', problem='amm mithral enc', N=49284.0, D=27.0, M=2.0, C=64.0, lut_work_coef=-1.0, n_reps=5, n_trials=10, trial_best_mult_ops_per_sec=[25590000.0, 25412000.0, 24037000.0, 25178000.0, 26576000.0]), cpp_timing_results(dataset_name='Caltech3x3', numeric_type=' f32', problem='amm mithral scan', N=49284.0, D=27.0, M=2.0, C=64.0, lut_work_coef=-1.0, n_reps=5, n_trials=10, trial_best_mult_ops_per_sec=[448290000.0, 382260000.0, 382260000.0, 380790000.0, 382260000.0])]
int8_data=[cpp_timing_results(dataset_name='Caltech3x3', numeric_type=' i8 ', problem='amm mithral nolut', N=49284.0, D=27.0, M=2.0, C=64.0, lut_work_coef=-1.0, n_reps=5, n_trials=10, trial_best_mult_ops_per_sec=[85168000.0, 85094000.0, 82531000.0, 82187000.0, 84729000.0]), cpp_timing_results(dataset_name='Caltech3x3', numeric_type=' i8 ', problem='amm mithral denselut', N=49284.0, D=27.0, M=2.0, C=64.0, lut_work_coef=-1.0, n_reps=5, n_trials=10, trial_best_mult_ops_per_sec=[73381000.0, 80575000.0, 80575000.0, 78773000.0, 80378000.0]), cpp_timing_results(dataset_name='Caltech3x3', numeric_type=' i8 ', problem='mithral lut dense', N=49284.0, D=27.0, M=2.0, C=64.0, lut_work_coef=-1.0, n_reps=5, n_trials=10, trial_best_mult_ops_per_sec=[19725000000.0, 16437000000.0, 10958000000.0, 10958000000.0, 10958000000.0]), cpp_timing_results(dataset_name='Caltech3x3', numeric_type=' i8 ', problem='amm mithral enc', N=49284.0, D=27.0, M=2.0, C=64.0, lut_work_coef=-1.0, n_reps=5, n_trials=10, trial_best_mult_ops_per_sec=[99120000.0, 101360000.0, 101670000.0, 102100000.0, 110320000.0]), cpp_timing_results(dataset_name='Caltech3x3', numeric_type=' i8 ', problem='amm mithral scan', N=49284.0, D=27.0, M=2.0, C=64.0, lut_work_coef=-1.0, n_reps=5, n_trials=10, trial_best_mult_ops_per_sec=[382260000.0, 382260000.0, 382260000.0, 365270000.0, 365270000.0])]

for data in [float32_data, int8_data]:
  print(data[0].dataset_name, data[0].numeric_type)
  print(*sorted(
          map(
            lambda r: (r.problem, '{:.2E}'.format(max(r.trial_best_mult_ops_per_sec))),
            data
          ),
          key=lambda t: float(t[1]),
          reverse=True
          ),
        sep='\n')
#new_throughput = max(map(lambda r: max(r.trial_best_mult_ops_per_sec), float32_data))
new_throughput = min(map(lambda r: max(r.trial_best_mult_ops_per_sec), int8_data)) #type of operation matters

#Compare mult speeds to Python, Pytorch, etc on same data
ncodebooks=ncodebooks[0]
lutconsts=lutconsts[0]
task=mithral_wrapped.mithral_amm_task_float(N,D,M, ncodebooks, lutconsts)
X=task.X
Q=task.Q
#task.output() #returns task.amm.out_mat, which is of type ColMatrix. returns Eigen::Matrix<T, Eigen::Dynamic, 1>;
num_dists=task.output().size
Y_hat=task.output().data #Think it's set to X@Q in scan, so exactly matches Y
#task.output().data() makes 
#task.output().size() makes 

s=timer()
Y=X@Q
e=timer()
old_t=e-s
old_throughput=(num_dists*(10**3))/old_t
print('{:.2E}'.format(old_throughput))

print("1-R^2: ", np.mean((Y_hat-Y)**2)/(np.mean(Y**2)-np.mean(Y)**2),
          f"New way Times Faster: {new_throughput/old_throughput}")

