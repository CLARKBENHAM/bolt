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
import matplotlib.pyplot as plt

from python import vq_amm
try:
  repo_path=os.path.join(os.path.dirname(__file__), "..")
except:
  #added except since when vscode auto-restarts after crashes, __file__ isn't defined
  repo_path = os.path.abspath('')[:os.path.abspath('').index('/bolt/') + 6]
sys.path.append(repo_path)
from cpp import mithral_wrapped

import functools
print = functools.partial(print, flush=True)

print(" Test pybind11 wrapped correctly")
assert 5 == mithral_wrapped.add(2, 3)
print("mithral ran")

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
ncodebooks= 32 #[2, 4, 8, 16, 32, 64]
lutconsts= -1 #[-1, 1, 2, 4]
N,D,M=10000, 512, 100
kCifar100TaskShape=mithral_wrapped.MatmulTaskShape(N,D,M, "Cifar100")
print("warn: using Cifar100")

#% Get raw C++ speeds
float32_profile = lambda : mithral_wrapped._profile_mithral(kCaltechTaskShape0, [ncodebooks], [lutconsts])
int8_profile = lambda :mithral_wrapped._profile_mithral_int8(kCaltechTaskShape0, [ncodebooks], [lutconsts])
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
#print(f'new (cpp) throughput {new_throughput:.2E} vs Old Throughput: {old_throughput:.2E}')

#% Speed of Python C++ Bindings
N,D,M = 4096, 64,128 #Python output size doesn't match input size
n,m=N,M
ncodebooks=2
task=mithral_wrapped.mithral_amm_task_float(N,D,M, ncodebooks, lutconsts)
#easy debugging
X= np.stack([np.array([i%16]*(D//2) + [(i%16) for j in range(D//2)]) for i in range(N)])
Q= np.stack([np.array([i%16]*(M//2) + [16 + i%16]*(M//2)) for i in range(D)])
X= np.stack([np.array([i%16]*(D)) for i in range(N)])
Q= np.stack([np.array([(i%16)/10]*(M//2) + [(i%16)]*(M//2)) for i in range(D)])
task.X=X
task.Q=Q
#X = task.X
#Q = task.Q

Y_hat1=np.array(task.output().data)
s=timer()
task.run_matmul(True)
#.output() Returns a memoryview of type ColMatrix which is Eigen::Matrix<T, Eigen::Dynamic, 1>;
Y_hat=np.asarray(task.output().data)
e=timer()
num_dists=Y_hat.size
print(f'time to run mithral with python bindings: {e-s:.7f}s',
      f'    Throughput: {num_dists/(e-s):.2E}/sec')
print(f"{100*np.sum(Y_hat1!=Y_hat)/Y_hat.size:.1f}% changed after encoding") 
unfitted_Y=Y_hat
print(np.asarray(task.output().data))

# Compare speeds to Python default (inaccurate since C++ binds don't learn params)
# print(type(X), type(Q), X.shape, Q.shape, np.mean(X), np.mean(Q))
s=timer()
Y=X@Q
e=timer()
old_t=e-s #in cpu sec's
old_throughput=(num_dists)/old_t
#print(f"New way Times Faster: {new_throughput/old_throughput}")
#print(f'new (cpp) time {new_time} vs old time {old_t}')
#print("old vs Mithral cpp 1-R^2: ",1-r2_score(Y, Y_hat))
#print(np.mean(Y), np.mean(Y_hat))
#print(X.shape, Q.shape)

#% Copy Python Learned values to C++
#hparams_dict = {'ncodebooks': 4*ncodebooks, 'lut_work_const': lutconsts} #increase codebooks with more outputs?
hparams_dict = {'ncodebooks': ncodebooks, 'lut_work_const': lutconsts}
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

print(f"current pid:    {os.getpid()}\n")
print()
#%% utils
def print_amm_ptrs(amm):
  print("amm shapes and [0][0]")
  print('centroids'      , amm.getCentroids().shape      , amm.getCentroids()[0][0])
  print('splitdims'      , amm.getSplitdims().shape      , amm.getSplitdims()[0][0])
  print('splitvals'      , amm.getSplitvals().shape      , amm.getSplitvals()[0][0])
  print('encode_scales'  , amm.getEncode_scales().shape  , amm.getEncode_scales()[0][0])
  print('encode_offsets' , amm.getEncode_offsets().shape , amm.getEncode_offsets()[0][0])
  print('idxs'           , amm.getIdxs().shape           , amm.getIdxs()[0][0])
  
  print("amm mins maxes") 
  print('centroids'      , np.min(amm.getCentroids(      ) ) , np.max(amm.getCentroids( ) ))
  print('splitdims'      , np.min(amm.getSplitdims(      ) ) , np.max(amm.getSplitdims( ) ))
  print('splitvals'      , np.min(amm.getSplitvals(      ) ) , np.max(amm.getSplitvals( ) ))
  print('encode_scales'  , np.min(amm.getEncode_scales(  ) ) , np.max(amm.getEncode_scales( ) ))
  print('encode_offsets' , np.min(amm.getEncode_offsets( ) ) , np.max(amm.getEncode_offsets( ) ))
  print('idxs'           , np.min(amm.getIdxs(           ) ) , np.max(amm.getIdxs( ) ))
  print(amm.out_offset_sum, amm.out_scale)
  
def reshape_split_lists(enc, att: str):
  #(ncodebooks x 4) for values for ncodebook subspaces to pick from the 4 levels 
  #aka ncodebooks by the 4 dims to split on for each output
  num_elem=enc.ncodebooks*4
  return np.array([
    getattr(i, att)
    for a in enc.splits_lists 
    for i in a]).reshape(enc.ncodebooks,4)

def extract_py_vars(est):
  #py est splitvals is jagged 3d (ncodebooks x 4 x [1,2,4,8]). Reshape to (ncodebooks x 16)
  raw_splitvals=[[v for i in a for v in i.vals]
                     for a in est.enc.splits_lists 
                     ]
  default_sz = max(map(len, raw_splitvals))
  num_pad = 2**math.ceil(np.log2(default_sz)) - default_sz
  ## i idx is 2^{i-1} value can split nodes for all the ncodebook subspaces
  #C++ expected 0 padded values out to 16 per BST of split values
  #TODO Is this right row/column order?
  #in C++ uses (ncodebooks, 32) if row major accesses
  # iterate over the 32 for a given codebook
  # [v1,v2,v2,v3,v3,v3,v3,v4,v4,v4,v4,v4,v4,v4,v4]
  # (nsplits, [1,2,4,8]) 
  #but the last values get padded to 16
  
  raw_splitvals_padded=np.array([np.pad(l, (1,0))
            for l in raw_splitvals
            ])
  #Python: (X-offset) * scale; C++: X*scale + offset; before comparing to these splitvals
  #WARN: these can overflow sometimes from int8 
  splitvals=(raw_splitvals_padded-128).clip(-128,127).astype(np.int8)
  #splitvals_raw [[[120  28 156  12  76 140 204   4  36  68 100 132 164 196 228   0]]
  
  encode_scales = reshape_split_lists(est.enc, 'scaleby').astype(np.float32)
  raw_encode_offset = reshape_split_lists(est.enc, 'offset').astype(np.float32) 
  return {
      #x/indexes
      "splitdims": reshape_split_lists(est.enc, 'dim').astype(np.uint32), #these are what's called idxs in paper?
      "splitvals": splitvals,
      "encode_scales": encode_scales, 
      "encode_offsets": raw_encode_offset*-encode_scales - 128, 
      #q/lut
      "centroids": est.enc.centroids.astype(np.float32),
      #"idxs": ,  #only need to set if we have a sparse matrix; idxs is used in mithral_lut_sparse always, but idxs are fine by default
      # if lut_work_const then c++ is row (ncodebooks,D) set to ncodebooks*[range(D)], else its ncodebooks*[sorted(permutation(range(D), nnz_per_centroid))]
      #Python encode_X (idxs?!) = offsets into raveled indxs + idxs=[[i%16]*ncodebooks for i in range(N)]), the offset is added to each idxs row
      #cpp out_offset_sum/out_scale is set by results at _compute_offsets_scale_from_mins_maxs, after done learning luts in mithral_lut_dense&mithral_lut_spares. no need to copy
      #   But then it's never used?
      "out_offset_sum": np.float32(est.offset),
      "out_scale":      np.float32(est.scale),
  }

print({k: v.shape if isinstance(v, np.ndarray) else v 
       for k,v in extract_py_vars(est).items()})

py_vars = extract_py_vars(est)
[c, d,v,eo,es, osum, oscale] = itemgetter('centroids', 'splitdims','splitvals', 'encode_offsets', 'encode_scales', 'out_offset_sum', 'out_scale')(py_vars)
print(py_vars, 'X: ', X, 'Q: ', Q, 'Y: ', Y, 'A_enc: ', est.A_enc, 'Luts: ', est.luts.reshape(est.luts.shape[0],-1), sep='\n')
#print(np.ravel(est.A_enc, order='f')[:48])
#print(np.ravel(est.luts.reshape(est.luts.shape[0],-1), order='c')[:48])

#copying keeps order for these, but are the luts and codes the same?
#print(est.A_enc, est.luts)
#print(list(map(lambda i: (np.min(i), np.max(i)), [est.A_enc, est.luts])))
#codes are >255 but c++ is uint8 type?

def copy_python_to_amm(py_est, amm):
  py_vars = extract_py_vars(py_est)
  [c, d,v,eo,es, osum, oscale] = itemgetter('centroids', 'splitdims','splitvals', 'encode_offsets', 'encode_scales', 'out_offset_sum', 'out_scale')(py_vars)

  #The centroid shape doens't matter? task (64,64) vs. est (4,16,64) 
  amm.setCentroidsCopyData(c)
  
  amm.setSplitdims(d)
  
  amm.setSplitvals(v)

  amm.setEncode_scales(es) #Do these get changed in training?
  amm.setEncode_offsets(eo)
  
  amm.out_offset_sum = osum
  #Python oscale is linear, C++ is exponent (wrong?)
  #C++ calculating scale
  #  float exponent = std::ceil(std::log2f(out_scale));
  #  out_scale = std::exp2(-exponent);  // reciprocal so we can use fma
  #  out_scale *= (255.f - 1e-10f);  // so max val is at most just under 255
  # Python
  #Python calculating scale 
  #  exponent = np.ceil(np.log2(gap))
  #  scale = 2 ** int(-exponent)  # scale is a power of 2, so can just shift
  #  scale *= (255.5 - 1e-10)  # so max val is at most 255
  #Python uses scale by dividing right before set value
  # C++ by imbedding change in luts:
  #  out_offsets[c] *= out_scale;
  
  amm.out_scale  = oscale
  #amm.setIdxs(.astype(int)) #only for non-dense
  
  #assert np.all(amm.getCentroids() == c) #shape wrong, return differently?
  assert np.all(np.ravel(amm.getCentroids()) == np.ravel(c)) 
  assert np.all(amm.getSplitdims() == d)
  #assert np.all(amm.getSplitvals() == v)
  assert np.all(amm.getEncode_scales()==es)
  assert np.all(amm.getEncode_offsets() == eo)
   
  #segfaults when py_est is changed; but I should be able to delete?
  #del py_est 

#%%
#compare dists function to make sure copied to c++ correctly

def dists_enc(self, X_enc, Q_luts, unquantize=True,
              offset=None, scale=None):
    X_enc = np.ascontiguousarray(X_enc)
    
    if unquantize:
        offset = self.total_lut_offset if offset is None else offset
        scale = self.scale_by if scale is None else scale

    all_dists = np.empty((len(Q_luts), len(X_enc)), dtype=np.float32)
    for i, lut in enumerate(Q_luts):
        centroid_dists = lut.ravel()[X_enc.ravel()]
        dists = centroid_dists.reshape(X_enc.shape)
        dists = dists.reshape(dists.shape[0], -1, self.upcast_every)
        # #upcast can be less than ncodebooks; in c++ you can't do the mutli-stage averaging more than 256 times, so if you had 1024 codebooks you'd seperate the averaging into 4 256 chunks and then average the final 4
        # #below is to have python mimic this upcast behaviour
        dists = dists.sum(2)
        dists = np.clip(dists, 0, 255).sum(axis=-1)
        if self.quantize_lut and unquantize:
            dists = (dists / scale) + offset
        all_dists[i] = dists
    return all_dists.T
  
def dists_enc_short(X_enc, Q_luts, 
            offset=0, scale=1,upcast_every=2):
  X_enc = np.ascontiguousarray(X_enc)
  
  all_dists = np.empty((len(Q_luts), len(X_enc)), dtype=np.float32)
  for i, lut in enumerate(Q_luts):
      centroid_dists = lut.ravel()[X_enc.ravel()]
      dists = centroid_dists.reshape(X_enc.shape)
      dists = np.clip(dists, 0, 255).sum(axis=-1)
      dists = (dists / scale) + offset
      
      all_dists[i] = dists
      if False: #slow
          assert(np.array_equal(dists,
                            [sum((centroid_dists[j*ncodebooks+c_ix] 
                          for c_ix in range(ncodebooks))
                          )/scale + offset for j in range(len(X_enc))]))
  return all_dists.T

def dists_enc_cpp(X_enc, Q_luts, 
            offset, scale, ncodebooks=2):
  "uses no Python specific syntax"
  Q_luts=Q_luts.reshape(Q_luts.shape[0],-1)
  n,_ = X_enc.shape
  m,_=Q_luts.shape
  dists_out=np.zeros(n*m)
  luts=np.ravel(Q_luts, order='c')
  codes=np.ravel(X_enc, order='f')
  for i in range(m):
    lut = luts[i * ncodebooks*16: (i+1)*ncodebooks*16]
    for j in range(n):
      dist = 0
      for code_ix in range(ncodebooks):
        dist += lut[codes[j + code_ix*n]]
      dists_out[i+j*m] = ((dist / scale) + offset)
  out = dists_out.reshape(n,m)
  return out

#Y_hat=dists_enc(est.enc, est.A_enc, est.luts, offset=est.offset, scale=est.scale)
#print("python hardcode using sum", 1-r2_score(Y, Y_hat))
#>>python hardcode using sum 7.626492367063253e-05

if False:
  #Ablations
  #If cast vals to int8 would py also be bad?
  est.fit(X,Q)
  py_vars = extract_py_vars(est)
  [c, d,v,eo,es, osum, oscale] = itemgetter('centroids', 'splitdims','splitvals', 'encode_offsets', 'encode_scales', 'out_offset_sum', 'out_scale')(py_vars)
  for code_ix, a in enumerate(est.enc.splits_lists):
    for row_ix, split in enumerate(a):
      split.vals = v[code_ix][2**row_ix:2**(row_ix+1)] #c++ is 1 indexed 
      split.offset = eo[code_ix][row_ix]
       
  est.A_enc = est.enc.encode_X(X)
  est.luts, est.offset, est.scale = est.enc.encode_Q(Q.T)
  est.fit(X,Q) #So est is in good state again by end
Y_hat=dists_enc_cpp(est.A_enc, est.luts, offset=est.offset, scale=est.scale)
print("cpp mimic in python", 1-r2_score(Y, Y_hat))
#%%
copy_python_to_amm(est, task.amm)
#task.amm.codes=est.A_enc
#task.amm.luts = est.luts.reshape(est.luts.shape[0],-1)
task.mithral_encode_only()
task.lut()
# #Hard setting codes and luts works
# good if C++ create's it's own luts
# <0 if C++ creates it's own codes
# <0 if C++ creates both luts and codes
task.amm.scan_test()
Y_hat=task.amm.out_mat
print("c++ copied to hardcode using sum", 1-r2_score(Y, Y_hat))
if r2_score(Y, Y_hat) < 0.99:
  plt.title(f"WARN BASIC COPYING BAD 1-R^2: {1-r2_score(Y, Y_hat):.4f}")
  plt.hist(Y.flatten(),bins=30,label='Y')
  plt.hist(Y_hat.flatten(),bins=30,label='Y_hat')
  plt.legend()
  plt.show()
#%% Trying hardcodes to debug, only copy luts and codes

def hardcode_copy(py_test, amm):
  copy_python_to_amm(est, amm) #not used for scan
  
  #Only for testing; in prod C++ must learn these vars itself
  #If these are only 4 bit codes, then ith columns should substract to be in range[0-16] not [i*16,(i+1)*16]
  h,w = est.A_enc.shape
  copy =est.A_enc #- np.array([[16*i for i in range(w)]]*h)
  amm.tmp_codes=copy 
  assert np.all(amm.tmp_codes==copy)
  amm.zip_bolt_colmajor_only() #format in C++
  #ix = o!=0
  #print('fraction unchanged: ', np.sum(amm.codes[ix] == o[ix])/o[ix].size) #didn't get changed?
  #assert(np.sum(amm.codes[ix] == o[ix])/o[ix].size < 0.2)
  
  #print(np.ravel(est.A_enc, order='f')[:32]) #ColMatrix
  #Codes are good: 
  #-exec p/d *(uint8_t *)&codes_ptr[4064]@32
  #$20 = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15}
  #-exec p/d *(uint8_t *)&codes_ptr[2*4096-32]@32
  #$21 = {16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31}
  
  amm.luts = est.luts.reshape(est.luts.shape[0],-1)
  assert np.all(np.ravel(amm.luts)==np.ravel(est.luts))
  #print(np.ravel(est.luts, order='c')[:32]) #RowMatrix
  #Luts are good: -exec p/d *(uint8_t *)&luts[127*32]@32
  #-exec p/d *(uint8_t *)&luts[127*32]@32
  #$22 = {0, 1, 2, 3, 4, 5, 6, 7, 7, 8, 9, 10, 11, 12, 13, 14, 0, 10, 19, 29, 39, 48, 58, 68, 77, 87, 97, 106, 116, 126, 135, 145}

old_codes = task.amm.codes
hardcode_copy(est, task.amm)
#task.scan()
task.amm.scan_test()
Y_hat=task.output()
print("Still only half?!", np.sum(Y_hat[:,:64]==0)/Y_hat[:,:64].size==0, np.sum(Y_hat[:,-64:]==0)/Y_hat[:,-64:].size==1, np.mean(Y_hat==0))
print("1-R^2 HARDCODE COPY: ",1-r2_score(Y, Y_hat))
print("1-R^2 First half HARDCODE COPY: ",1-r2_score(Y[:,:64], Y_hat[:,:64]))
task.scan()


#%%  Copy Python to C++
print_amm_ptrs(task.amm)
print("copying python")
copy_python_to_amm(est, task.amm)
print("copied python")
print_amm_ptrs(task.amm)

print("starting:      task.run_matmul(True)")
s=timer()
task.run_matmul(True)
#.output() Returns a memoryview of type ColMatrix which is Eigen::Matrix<T, Eigen::Dynamic, 1>;
Y_hat=task.output()#==task.amm.out_mat
#336:mithral.hpp; out_scale calcuated by 255*2^-math.ceil(log2(max_scale_val))
# out_offset_sum is sum of min offset across each codebook
#Y_hat=Y_hat/ task.amm.out_scale #((255 - 1e-10) * 2**math.floor(math.log2(task.amm.out_scale)))
#Y_hat+=task.amm.out_offset_sum #is this needed?
e=timer()
num_dists=Y_hat.size
print(f'time to run mithral with python bindings: {e-s:.7f}s',
      f'Throughput: {num_dists/(e-s):.2E}/sec',
      f'Python Throughtput {old_throughput:.2E}')
print(f"{100*np.sum(unfitted_Y!=Y_hat)/Y_hat.size:.1f}% changed after encoding. Num Not 0: {np.sum(Y_hat!=0)}") 
print("1-R^2: ",1-r2_score(Y, Y_hat))
print(f'new time {e-s} vs old time {old_t}')
Y_hat=est(X,Q)
#why true? np.all(task.output()[3104:,50:]==0)
#assert(np.all(task.amm.codes == est.A_enc)) #false why?
#assert(np.all(task.amm.luts.shape==est.luts.shape)) #wrong dims

#%% Scrap
copy_python_to_amm(est, task.amm)
for i in range(99):
  c, *_ = itemgetter('centroids', 'splitdims','splitvals', 'encode_offsets', 'encode_scales', 'out_offset_sum', 'out_scale')(py_vars)
  task.amm.setCentroidsCopyData(c)
  #task.amm.setCentroids(c)
  task.run_matmul(True)
  print(i) 
  #always stops at 54?

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

#mithral_wrapped.mithral_amm_float(N,D,M, ncodebooks,centroids, splitdims, splitvals, encode_scales, encode_offsets, idxs, nnz_per_centroid)

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
#  #del py_est ``
