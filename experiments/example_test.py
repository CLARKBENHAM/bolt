#%%
# Run the C++ version with pybind11 wrappings

import re
import os
import sys
import io
import math
from contextlib import contextmanager
import ctypes
import tempfile
from collections import namedtuple
from timeit import default_timer as timer
import numpy as np
from sklearn.metrics import r2_score
from operator import itemgetter

from python import vq_amm
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from cpp import mithral_wrapped

# Test pybind11 wrapped correctly
assert 5 == mithral_wrapped.add(2, 3)


#So C++ code's printout redirected to python
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

cpp_timing_results=namedtuple('cpp_timing_results', "dataset_name numeric_type problem N D M C lut_work_coef n_reps n_trials trial_best_mult_ops_per_sec trial_best_sec_times")

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
    #mithral returns in ms; convert to seconds
    trial_best_sec_times=[float(re.search('\d+\.\d+', i).group(0))/1000
          for i in results[::2]]
    return cpp_timing_results(dataset_name.strip(),
                              numeric_type.strip(),
                              problem.strip(),
                              N,D,M,C,lut_work_coef,
                              n_reps,
                              n_trials,
                              trial_best_mult_ops_per_sec, 
                              trial_best_sec_times)
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

def test_pybinding_worked(kCaltechTaskShape0):
  #kCaltechTaskShape0.name=name #if I assigned to it then name gets corrupted/errorrs
  print(type(kCaltechTaskShape0))
  print('N, D,M: ', kCaltechTaskShape0.N, kCaltechTaskShape0.D, kCaltechTaskShape0.M)
  print('name: ', kCaltechTaskShape0.name) 
  print(kCaltechTaskShape0, name)
  mithral_wrapped.printNameMatmul(kCaltechTaskShape0) 

#Task Constants
name = "Caltech3x3" #'cifar100' 
#N,D,M = (224 - 3 + 1) * (224 - 3 + 1), 3 * (3 * 3), 2 
N,D,M= 49312,27,2 #dimensions of X and Q above produces. These stay
kCaltechTaskShape0=mithral_wrapped.MatmulTaskShape(N,D,M, name)
ncodebooks= [16] #[2, 4, 8, 16, 32, 64]
lutconsts= [-1] #[-1, 1, 2, 4]
N,D,M=10000, 512, 100
kCifar100TaskShape=mithral_wrapped.MatmulTaskShape(N,D,M, "Cifar100")
kCaltechTaskShape0=kCifar100TaskShape
print("warn: using Cifar100")

#% Get raw C++ speeds
float32_profile = lambda : mithral_wrapped._profile_mithral(kCaltechTaskShape0, ncodebooks, lutconsts)
int8_profile = lambda :mithral_wrapped._profile_mithral_int8(kCaltechTaskShape0, ncodebooks, lutconsts)
##NOTE: The stdout redirects don't work in repl
#float32_data = run_cpp_timing(float32_profile)
#int8_data = run_cpp_timing(int8_profile)
#print("float32_data=", float32_data,'\nint8_data=', int8_data)

##Caltech3x3
#float32_data= [cpp_timing_results(dataset_name='Caltech3x3', numeric_type='f32', problem='amm mithral nolut', N=49284.0, D=27.0, M=2.0, C=64.0, lut_work_coef=-1.0, n_reps=5, n_trials=10, trial_best_mult_ops_per_sec=[20745000.0, 22517000.0, 22724000.0, 17665000.0, 15853000.0], trial_best_sec_times=[0.0047539999999999995, 0.00438, 0.00434, 0.005583, 0.006221]), cpp_timing_results(dataset_name='Caltech3x3', numeric_type='f32', problem='amm mithral denselut', N=49284.0, D=27.0, M=2.0, C=64.0, lut_work_coef=-1.0, n_reps=5, n_trials=10, trial_best_mult_ops_per_sec=[19776000.0, 21781000.0, 22830000.0, 22404000.0, 21892000.0], trial_best_sec_times=[0.004987, 0.004528, 0.00432, 0.0044020000000000005, 0.004505]), cpp_timing_results(dataset_name='Caltech3x3', numeric_type='f32', problem='mithral lut dense', N=49284.0, D=27.0, M=2.0, C=64.0, lut_work_coef=-1.0, n_reps=5, n_trials=10, trial_best_mult_ops_per_sec=[16437000000.0, 16437000000.0, 16437000000.0, 16437000000.0, 16437000000.0], trial_best_sec_times=[6e-06, 6e-06, 6e-06, 6e-06, 6e-06]), cpp_timing_results(dataset_name='Caltech3x3', numeric_type='f32', problem='amm mithral enc', N=49284.0, D=27.0, M=2.0, C=64.0, lut_work_coef=-1.0, n_reps=5, n_trials=10, trial_best_mult_ops_per_sec=[25089000.0, 26691000.0, 24406000.0, 26727000.0, 27057000.0], trial_best_sec_times=[0.0039310000000000005, 0.003695, 0.004041, 0.00369, 0.003645]), cpp_timing_results(dataset_name='Caltech3x3', numeric_type='f32', problem='amm mithral scan', N=49284.0, D=27.0, M=2.0, C=64.0, lut_work_coef=-1.0, n_reps=5, n_trials=10, trial_best_mult_ops_per_sec=[448290000.0, 448290000.0, 382260000.0, 382260000.0, 382260000.0], trial_best_sec_times=[0.00022, 0.00022, 0.000258, 0.000258, 0.000258])] 
#int8_data= [cpp_timing_results(dataset_name='Caltech3x3', numeric_type='i8', problem='amm mithral nolut', N=49284.0, D=27.0, M=2.0, C=64.0, lut_work_coef=-1.0, n_reps=5, n_trials=10, trial_best_mult_ops_per_sec=[98231000.0, 85021000.0, 84511000.0, 82808000.0, 85463000.0], trial_best_sec_times=[0.001004, 0.00116, 0.001167, 0.001191, 0.0011539999999999999]), cpp_timing_results(dataset_name='Caltech3x3', numeric_type='i8', problem='amm mithral denselut', N=49284.0, D=27.0, M=2.0, C=64.0, lut_work_coef=-1.0, n_reps=5, n_trials=10, trial_best_mult_ops_per_sec=[74321000.0, 79344000.0, 80906000.0, 73931000.0, 80641000.0], trial_best_sec_times=[0.001327, 0.0012430000000000002, 0.001219, 0.0013340000000000001, 0.0012230000000000001]), cpp_timing_results(dataset_name='Caltech3x3', numeric_type='i8', problem='mithral lut dense', N=49284.0, D=27.0, M=2.0, C=64.0, lut_work_coef=-1.0, n_reps=5, n_trials=10, trial_best_mult_ops_per_sec=[16437000000.0, 16437000000.0, 16437000000.0, 16437000000.0, 16437000000.0], trial_best_sec_times=[6e-06, 6e-06, 6e-06, 6e-06, 6e-06]), cpp_timing_results(dataset_name='Caltech3x3', numeric_type='i8', problem='amm mithral enc', N=49284.0, D=27.0, M=2.0, C=64.0, lut_work_coef=-1.0, n_reps=5, n_trials=10, trial_best_mult_ops_per_sec=[110690000.0, 112710000.0, 110320000.0, 110440000.0, 110940000.0], trial_best_sec_times=[0.000891, 0.000875, 0.000894, 0.000893, 0.000889]), cpp_timing_results(dataset_name='Caltech3x3', numeric_type='i8', problem='amm mithral scan', N=49284.0, D=27.0, M=2.0, C=64.0, lut_work_coef=-1.0, n_reps=5, n_trials=10, trial_best_mult_ops_per_sec=[382260000.0, 448290000.0, 448290000.0, 448290000.0, 448290000.0], trial_best_sec_times=[0.000258, 0.00022, 0.00022, 0.00022, 0.00022])]
##Cifar100
float32_data= [cpp_timing_results(dataset_name='Cifar100', numeric_type='f32', problem='amm mithral nolut', N=10000.0, D=512.0, M=100.0, C=64.0, lut_work_coef=-1.0, n_reps=5, n_trials=10, trial_best_mult_ops_per_sec=[317970000.0, 319590000.0, 315070000.0, 307520000.0, 300510000.0], trial_best_sec_times=[0.00315, 0.003134, 0.003179, 0.003257, 0.003333]), cpp_timing_results(dataset_name='Cifar100', numeric_type='f32', problem='amm mithral denselut', N=10000.0, D=512.0, M=100.0, C=64.0, lut_work_coef=-1.0, n_reps=5, n_trials=10, trial_best_mult_ops_per_sec=[92220000.0, 103780000.0, 101910000.0, 98865000.0, 99493000.0], trial_best_sec_times=[0.010861, 0.009651, 0.009828, 0.010131, 0.010067]), cpp_timing_results(dataset_name='Cifar100', numeric_type='f32', problem='mithral lut dense', N=10000.0, D=512.0, M=100.0, C=64.0, lut_work_coef=-1.0, n_reps=5, n_trials=10, trial_best_mult_ops_per_sec=[161840000.0, 161860000.0, 161760000.0, 155960000.0, 138760000.0], trial_best_sec_times=[0.006189, 0.006188, 0.0061920000000000005, 0.006422, 0.007218]), cpp_timing_results(dataset_name='Cifar100', numeric_type='f32', problem='amm mithral enc', N=10000.0, D=512.0, M=100.0, C=64.0, lut_work_coef=-1.0, n_reps=5, n_trials=10, trial_best_mult_ops_per_sec=[1157900000.0, 1174200000.0, 1253600000.0, 1309300000.0, 1323100000.0], trial_best_sec_times=[0.000865, 0.000853, 0.000799, 0.0007650000000000001, 0.000757]), cpp_timing_results(dataset_name='Cifar100', numeric_type='f32', problem='amm mithral scan', N=10000.0, D=512.0, M=100.0, C=64.0, lut_work_coef=-1.0, n_reps=5, n_trials=10, trial_best_mult_ops_per_sec=[478550000.0, 478550000.0, 455690000.0, 430430000.0, 431170000.0], trial_best_sec_times=[0.002093, 0.002093, 0.002198, 0.002327, 0.002323])] 
int8_data= [cpp_timing_results(dataset_name='Cifar100', numeric_type='i8', problem='amm mithral nolut', N=10000.0, D=512.0, M=100.0, C=64.0, lut_work_coef=-1.0, n_reps=5, n_trials=10, trial_best_mult_ops_per_sec=[438340000.0, 440070000.0, 439680000.0, 350210000.0, 352550000.0], trial_best_sec_times=[0.002285, 0.002276, 0.002278, 0.0028599999999999997, 0.002841]), cpp_timing_results(dataset_name='Cifar100', numeric_type='i8', problem='amm mithral denselut', N=10000.0, D=512.0, M=100.0, C=64.0, lut_work_coef=-1.0, n_reps=5, n_trials=10, trial_best_mult_ops_per_sec=[98081000.0, 103930000.0, 105540000.0, 101000000.0, 103330000.0], trial_best_sec_times=[0.010212, 0.009637, 0.00949, 0.009917, 0.009693]), cpp_timing_results(dataset_name='Cifar100', numeric_type='i8', problem='mithral lut dense', N=10000.0, D=512.0, M=100.0, C=64.0, lut_work_coef=-1.0, n_reps=5, n_trials=10, trial_best_mult_ops_per_sec=[153810000.0, 151070000.0, 144700000.0, 142760000.0, 151780000.0], trial_best_sec_times=[0.0065119999999999996, 0.00663, 0.006921999999999999, 0.007016, 0.006599]), cpp_timing_results(dataset_name='Cifar100', numeric_type='i8', problem='amm mithral enc', N=10000.0, D=512.0, M=100.0, C=64.0, lut_work_coef=-1.0, n_reps=5, n_trials=10, trial_best_mult_ops_per_sec=[5414100000.0, 5823300000.0, 5033200000.0, 5058600000.0, 5384900000.0], trial_best_sec_times=[0.000185, 0.00017199999999999998, 0.000199, 0.00019800000000000002, 0.000186]), cpp_timing_results(dataset_name='Cifar100', numeric_type='i8', problem='amm mithral scan', N=10000.0, D=512.0, M=100.0, C=64.0, lut_work_coef=-1.0, n_reps=5, n_trials=10, trial_best_mult_ops_per_sec=[434530000.0, 427670000.0, 435670000.0, 419780000.0, 373870000.0], trial_best_sec_times=[0.002305, 0.002342, 0.002299, 0.002386, 0.002679])]

int8_num_dists = [rate*time for r in int8_data for rate,time in zip(r.trial_best_mult_ops_per_sec, r.trial_best_sec_times)]
#sometimes a little off the 98624 from the output matrix size
#print(max(int8_num_dists), min(int8_num_dists), np.mean(int8_num_dists), np.std(int8_num_dists))

for data in [float32_data, int8_data]:
  print(data[0].dataset_name, data[0].numeric_type)
  print(*sorted(
          map(
            lambda r: (r.problem,
                       '{:.2E}'.format(max(r.trial_best_mult_ops_per_sec)),
                       min(r.trial_best_sec_times),
                      ),
            data
          ),
          key=lambda t: t[2],
          ),
        sep='\n')
#new_throughput = max(map(lambda r: max(r.trial_best_mult_ops_per_sec), float32_data))
new_throughput = min(map(lambda r: max(r.trial_best_mult_ops_per_sec), int8_data)) #type of operation matters
new_time = max(map(lambda r: min(r.trial_best_sec_times), int8_data)) #type of operation matters


#% Speed of Python C++ Bindings
task=mithral_wrapped.mithral_amm_task_float(N,D,M, ncodebooks[0], lutconsts[0])
X=task.X
Q=task.Q
Y_hat1=np.array(task.output().data)
s=timer()
task.run_matmul(True)
#.output() Returns a memoryview of type ColMatrix which is Eigen::Matrix<T, Eigen::Dynamic, 1>;
Y_hat=np.asarray(task.output().data)
e=timer()
num_dists=Y_hat.size
print(f'time to run mithral with python bindings: {e-s:.7f}s',
      f'Throughput: {num_dists/(e-s):.2E}/sec')
print(f"{100*np.sum(Y_hat1!=Y_hat)/Y_hat.size:.1f}% changed after encoding") 
unfitted_Y=Y_hat

# Compare speeds to Python default (inaccurate since C++ binds don't learn params)
# print(type(X), type(Q), X.shape, Q.shape, np.mean(X), np.mean(Q))
s=timer()
Y=X@Q
e=timer()
old_t=e-s #in cpu sec's
old_throughput=(num_dists)/old_t
print(f'new_throughput {new_throughput:.2E} vs Old Throughput: {old_throughput:.2E}')
print(f'new time {new_time} vs old time {old_t}')
print(f"New way Times Faster: {new_throughput/old_throughput}")

print("1-R^2: ",1-r2_score(Y, Y_hat))
print(r2_score(Y, Y_hat1), r2_score(Y, Y_hat))
print(np.mean(Y), np.mean(Y_hat))
print(X.shape, Q.shape)


#% Copy Python Learned values to C++
#hparams_dict = {'ncodebooks': 4*ncodebooks[0], 'lut_work_const': lutconsts[0]} #increase codebooks with more outputs?
hparams_dict = {'ncodebooks': ncodebooks[0], 'lut_work_const': lutconsts[0]}
est = vq_amm.MithralMatmul(**hparams_dict)
#Num centroids is fixed at 16; so can always fit in L1?    
est.fit(X,Q)
s=timer()
Y_hat=est(X, Q) #makes est. but equivalent to est(None, Q)
e=timer()
print(f"Python implementation of Mithral faster by: {old_t/(e-s)}, 1-R^2: {1-r2_score(Y, Y_hat)}")
est_attr=['A_enc',
  'enc',
  'fit',
  'get_params',
  'get_speed_metrics',
  'lut_work_const',
  'luts',
  'ncentroids',
  'ncodebooks',
  'offset',
  'predict',
  'reset_for_new_task',
  'scale',
  'set_A',
  'set_B']
#for t in est_attr:
#  print(t, est.__getattribute__(t))
py_est = est
#%%
#import copy
def print_amm_ptrs(amm):
  print(amm.getCentroids().shape      , amm.getCentroids()[0][0])
  print(amm.getSplitdims().shape      , amm.getSplitdims()[0][0])
  print(amm.getSplitvals().shape      , amm.getSplitvals()[0][0])
  print(amm.getEncode_scales().shape  , amm.getEncode_scales()[0][0])
  print(amm.getEncode_offsets().shape , amm.getEncode_offsets()[0][0])
  print(amm.getIdxs().shape           , amm.getIdxs()[0][0])
  
  print(np.min(amm.getCentroids(      ) ) , np.max(amm.getCentroids( ) ))
  print(np.min(amm.getSplitdims(      ) ) , np.max(amm.getSplitdims( ) ))
  print(np.min(amm.getSplitvals(      ) ) , np.max(amm.getSplitvals( ) ))
  print(np.min(amm.getEncode_scales(  ) ) , np.max(amm.getEncode_scales( ) ))
  print(np.min(amm.getEncode_offsets( ) ) , np.max(amm.getEncode_offsets( ) ))
  print(np.min(amm.getIdxs(           ) ) , np.max(amm.getIdxs( ) ))


def reshape_split_lists(enc, att: str):
  #(ncodebooks x 4) for values for ncodebook subspaces to pick from the 4 levels 
  #aka ncodebooks by the 4 dims to split on for each output
  num_elem=enc.ncodebooks*4
  return np.array([
    getattr(i, att)
    for a in enc.splits_lists 
    for i in a]).reshape(enc.ncodebooks,4)

def extract_py_vars(est):
  raw_splitvals=[[v for i in a for v in i.vals]
                     for a in est.enc.splits_lists 
                     ]
  default_sz = max(map(len, raw_splitvals))
  num_pad = 2**math.ceil(np.log2(default_sz)) - default_sz
  ## 3d: ncodebooks x[1, 2, 4, 8], i idx is 2^{i-1} value can split nodes for all the ncodebook subspaces
  #C++ expected 0 padded values out to 16 per BST of split values
  #TODO Is this right row/column order?
  splitvals=np.array([np.pad(l, (0,num_pad))
            for l in raw_splitvals
            ])
  return {
      #x/indexes
      "splitdims": reshape_split_lists(est.enc, 'dim').astype(np.uint32),
      "splitvals": splitvals,
      "encode_scales":reshape_split_lists(est.enc, 'scaleby').astype(np.float32),
      "encode_offsets":reshape_split_lists(est.enc, 'offset').astype(np.float32),
  	  #q/lut
      "centroids": est.enc.centroids,
      #"idxs": , #only with a sparse LUT
      #cpp out_offset_sum/scale is set by results at _compute_offsets_scale_from_mins_maxs, after done learning luts in mithral_lut_dense&mithral_lut_spares. no need to copy
      #So do you need to modify c output by scale/sum to be correct?
      #py 
      "out_offset_sum":est.offset,
      "out_scale": est.scale,
  }

def copy_python_to_amm(py_est, amm):
  py_vars = extract_py_vars(py_est)
  [c, d,v,eo,es] = itemgetter('centroids', 'splitdims','splitvals', 'encode_offsets', 'encode_scales')(py_vars)
  amm.setCentroids(c)
  
  amm.setSplitdims(d)
  
  amm.setSplitvals(v)
  
  amm.setEncode_scales(es)
  amm.setEncode_offsets(eo)
  #amm.setIdxs(.astype(int))
  
  #assert np.all(amm.getSplitdims() == d)
  #assert np.all(amm.getEncode_scales()==es)
  #assert np.all(amm.getEncode_offsets() == eo)
  
  #segfaults when py_est is changed; but I should be able to delete?
  #del py_est 

print_amm_ptrs(task.amm)
copy_python_to_amm(est, task.amm)
print_amm_ptrs(task.amm)

s=timer()
task.run_matmul(True)
#.output() Returns a memoryview of type ColMatrix which is Eigen::Matrix<T, Eigen::Dynamic, 1>;
Y_hat=np.asarray(task.output().data)
#336:mithral.hpp; out_scale calculated by 255*2^-math.ceil(log2(max_scale_val))
# out_offset_sum is sum of min offset across each codebook
Y_hat=Y_hat/ task.amm.out_scale #((255 - 1e-10) * 2**math.floor(math.log2(task.amm.out_scale)))
Y_hat+=task.amm.out_offset_sum
e=timer()
num_dists=Y_hat.size
print(f'time to run mithral with python bindings: {e-s:.7f}s',
      f'Throughput: {num_dists/(e-s):.2E}/sec',
      f'Python Throughtput {old_throughput:.2E}')
print(f"{100*np.sum(unfitted_Y!=Y_hat)/Y_hat.size:.1f}% changed after encoding") 
print(f'new time {e-s} vs old time {old_t}')
print("1-R^2: ",1-r2_score(Y, Y_hat))
print({k: v.shape if isinstance(v, np.ndarray) else v 
       for k,v in extract_py_vars(est).items()})
#%%
copy_python_to_amm(est, task.amm)
for i in range(99):
  task.amm.setCentroidsCopyData(est.enc.centroids)
  #task.amm.setCentroids(est.enc.centroids)
  task.run_matmul(True)
  print(i)
#%%
print({k: v.shape if isinstance(v, np.ndarray) else v 
       for k,v in extract_py_vars(est).items()})
#fitted = ['ncodebooks', 'ncentroids', 'A_enc', 'luts', 'offset', 'scale'] #works to copy in python(?)
#things you're supposed to learn per mithral_amm_task constructor
#        amm(N_padded, D, M, ncodebooks, centroids.data(),
#            splitdims.data(), splitvals.data(),
#            encode_scales.data(), encode_offsets.data(),
#            idxs.data(), nnz_per_centroid), //mithral_amm type is def'd in mithral.hpp

# #set on task.amm since the task is already made(?)
# #copy from the python which has been trained
#task.N_padded         = 
#task.centroids        = est.enc.centroids
#task.nsplits          = 
#task.splitdims        = 
#task.splitvals        = 
#task.encode_scales    = est.enc.scales
#task.encode_offsets   = est.offset
#task.nnz_per_centroid = 
#task.idxs             = 

#print(f'''\n\ntask:
#task.N_padded         = {task.N_padded         }
#task.centroids        = {task.centroids        }
#task.nsplits          = {task.nsplits          }
#task.splitdims        = {task.splitdims        }
#task.splitvals        = {task.splitvals        }
#task.encode_scales    = {task.encode_scales    }
#task.encode_offsets   = {task.encode_offsets   }
#task.nnz_per_centroid = {task.nnz_per_centroid }
#task.idxs             = {task.idxs             }
#''')
#print(f'''\n\namm:
#task.amm.N                = {task.amm.N                               }
#task.amm.D                = {task.amm.D                               }
#task.amm.M                = {task.amm.M                               }
#task.amm.ncodebooks       = {task.amm.ncodebooks                      }
#task.amm.centroids        = {task.amm.centroids                       }
#task.amm.splitdims        = {task.amm.splitdims                       }
#task.amm.splitvals        = {task.amm.splitvals                       }
#task.amm.encode_scales    = {task.amm.encode_scales                   }
#task.amm.encode_offsets   = {task.amm.encode_offsets                  }
#task.amm.idxs             = {task.amm.idxs                            }
#task.amm.nnz_per_centroid = {task.amm.nnz_per_centroid                }
#task.amm.tmp_codes        = {task.amm.tmp_codes                       }
#task.amm.codes            = {task.amm.codes                           }
#task.amm.tmp_luts_f32     = {task.amm.tmp_luts_f32                    }
#task.amm.luts             = {task.amm.luts                            }
#task.amm.out_offset_sum   = {task.amm.out_offset_sum                  }
#task.amm.out_scale        = {task.amm.out_scale                       }
#task.amm.out_mat          = {task.amm.out_mat                         }
#''')

#Python LUTs are the wrong size
print(task.amm.codes.shape, task.amm.luts.shape)
print(est.A_enc.shape, est.luts.shape)
#(49312, 16) (49312, 256)
#(49312, 16) (2, 16, 16)
print(np.min(est.luts), np.max(est.luts))
#0 168

#Just set intermidiate values needed for scan(?)
assert task.amm.codes.shape==est.A_enc.shape
task.amm.codes=est.A_enc
assert task.amm.luts.shape==est.luts.shape #fails, prime factor's diff by odd amount
task.amm.luts=est.luts

#Test Copied correctly
Q2=Q+np.random.rand(*Q.shape)*np.mean(Q)/10
#%% Scrap

mithral_wrapped.mithral_amm_float(N,D,M, ncodebooks,centroids, splitdims, splitvals, encode_scales, encode_offsets, idxs, nnz_per_centroid)

#Do you need the constructor params or just the intermedia values?

#def copy_python_to_amm_safe(py_est, amm):
#  #py_est = copy.deepcopy(py_e)
#  amm.setCentroidsCopyData(py_est.enc.centroids)
#  #64x1
#  reshape_split_lists=lambda att: np.array([
#    getattr(i, att)
#    for a in py_est.enc.splits_lists 
#    for i in a]).reshape(64,1)
#  print('hi')
#  d=reshape_split_lists('dim').astype(np.uint32)
#  amm.setSplitdimsCopyData(d)
#  assert np.all(amm.getSplitdims() == d)
#  print('ran')
#  
#  ## len 240 in 3d: 16x[1, 2, 4, 8]
#  #amm.setSplitvals([v 
#  #                       for a in py_est.enc.splits_lists 
#  #                       for i in a
#  #                       for v in i.vals])
#  #amm.setSplitvals(reshape_split_lists('vals'))
#  
#  s=reshape_split_lists('scaleby').astype(np.float32)
#  amm.setEncode_scalesCopyData(s)
#  assert np.all(amm.getEncode_scales()==s)
#  
#  o=reshape_split_lists('offset').astype(np.float32)
#  amm.setEncode_offsetsCopyData(o)
#  assert np.all(amm.getEncode_offsets() == o)
#  #amm.setIdxsCopyData(.astype(int))
#  
#  #segfaults when py_est is changed; but I should be able to delete?
#  #del py_est 
