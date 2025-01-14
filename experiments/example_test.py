#%%
# Run the C++ version with pybind11 wrappings
# Collection of Scrap code

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

assert 5 == mithral_wrapped.add(2, 3)
print(os.getpid())

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
N,D,M = 65536, 64,128 
#N,D,M = 4096, 64,128 
n,m=N,M
ncodebooks=2
task=mithral_wrapped.mithral_amm_task_float(N,D,M, ncodebooks, lutconsts)
#easy debugging

X= np.stack([np.array([i%16]*(D//2) + [(i%16) for j in range(D//2)]) for i in range(N)])
Q= np.stack([np.array([i%16]*(M//2) + [16 + i%16]*(M//2)) for i in range(D)])
#X[:,:16]*=32 #1-R^2 for cpp: 0.33; Python version: 0.12 only when multiply 1st subspace by 32 no negative changes
X[:,::3]*=-1
#X[:,::2]*=-1 #If I do all 3 of these then 1-R^2=0.11 on both Py and CPP
#X*=-1 
#X=X+np.random.randint(0,10,[N,D])/3 /1 R^2<0, /3 R^2=0.95, /10 R^2=0.98
#Q=Q+np.random.randint(0,10,[D,M])/3
#X=X.astype(float)/10
#Q=Q.astype(float)/10
#X= np.stack([np.array([i%16]*(D)) for i in range(N)])
#Q= np.stack([np.array([(i%16)/10]*(M//2) + [(i%16)]*(M//2)) for i in range(D)])
#X = task.X
#Q = task.Q
task.X=X
task.Q=Q

Y_hat1=np.array(task.output().data)
s=timer()
task.run_matmul(True) 
#.output() Returns a memoryview of type ColMatrix which is Eigen::Matrix<T, Eigen::Dynamic, 1>;
Y_hat =np.array(task.amm.scan_ret_col_order_upcast(), copy=False)
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
print(f"Python implementation of Mithral faster by: {old_t/(e-s)}, 1-R^2: {1-r2_score(Y, Y_hat)}, warn will be off since we modified python; change back to process_x")
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
  c = est.enc.centroids.astype(np.float32)
  #reshaped_centroids = np.ravel(np.apply_along_axis(lambda cbook : np.ravel(cbook, order='f'), 0,c))
  #reshaped_centroids = np.concatenate((np.ravel(c[0], order='f'), np.ravel(c[1], order='f')))
  reshaped_centroids=np.concatenate(list(map(lambda cbook_ix : np.ravel(c[cbook_ix], order='f'), range(len(c)))))
  return {
      #x/indexes
      "splitdims": reshape_split_lists(est.enc, 'dim').astype(np.uint32), #these are what's called idxs in paper?
      "splitvals": splitvals,
      "encode_scales": encode_scales, 
      "encode_offsets": raw_encode_offset*-encode_scales - 128, 
      #q/lut
      "centroids": reshaped_centroids,
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

  amm.setCentroidsCopyData(c)
  
  amm.setSplitdims(d)
  
  amm.setSplitvals(v)

  amm.setEncode_scales(es)
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

#%%  Copy Python Params to C++, which creates luts/codes and generates Y_hat
print(f"starting:      copying {os.getpid()}")
copy_python_to_amm(est, task.amm)
task.amm.out_mat = np.zeros(task.amm.out_mat.shape)
print("succesful copy")
s=timer()
task.run_matmul(True)
#If use uint8 Mithral Output then need to cast away first
Y_hat=(task.amm.out_mat.astype(np.float32)*ncodebooks/task.amm.out_scale) + task.amm.out_offset_sum #by avg
#Y_hat=(task.amm.out_mat/task.amm.out_scale) + task.amm.out_offset_sum #by sum
#Y_hat=task.amm.out_mat
e=timer()
num_dists=Y_hat.size
print("1-R^2: ",1-r2_score(Y, Y_hat))
print(f'C++ {old_t/(e-s)} times faster; new time {e-s} vs old time {old_t}')
print(f'time to run mithral with python bindings: {e-s:.7f}s',
      f'Throughput: {num_dists/(e-s):.2E}/sec',
      f'Python Throughtput {old_throughput:.2E}')

#print(np.unique(task.amm.out_mat),
#      np.unique(((Y-task.amm.out_offset_sum)*task.amm.out_scale).astype(int)))

#%%  Debug Functions
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
            offset=0, scale=1):
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
            offset=0, scale=1, ncodebooks=2):
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

#Setup Python to be how C++ expects it
if False: #Make it false when debugging
  #tests Do Python the way C++ is done and make sure values are correct
  #Edit clusterize.py:assignments_from_multisplits to use splits.preprocess_x_like_cpp
  est.fit(X,Q)
  py_vars = extract_py_vars(est)
  [c, d,v,eo,es, osum, oscale] = itemgetter('centroids', 'splitdims','splitvals', 'encode_offsets', 'encode_scales', 'out_offset_sum', 'out_scale')(py_vars)
  for code_ix, a in enumerate(est.enc.splits_lists):
    for row_ix, split in enumerate(a):
      split.vals = v[code_ix][2**row_ix:2**(row_ix+1)] #c++ is 1 indexed 
      split.offset = eo[code_ix][row_ix] 
      
  #These are a little off now, but C++ is still good 
  est.A_enc = est.enc.encode_X(X)
  est.luts, est.offset, est.scale = est.enc.encode_Q(Q.T)
  Y_hat=dists_enc(est.enc, est.A_enc, est.luts, offset=est.offset, scale=est.scale)
  print("cpp mimic in python; to int8", 1-r2_score(Y, Y_hat))
  print("Vars how C++ should make them")
  print(py_vars, 'X: ', X, 'Q: ', Q, 'Y: ', Y, 'A_enc: ', est.A_enc, 'Luts: ', est.luts.reshape(est.luts.shape[0],-1), sep='\n')
  est.fit(X,Q) #So est is in good state again by end

Y_hat_py=dists_enc_cpp(est.A_enc, est.luts, offset=est.offset, scale=est.scale, ncodebooks=ncodebooks)
print("cpp mimic in python", 1-r2_score(Y, Y_hat_py))
copy_python_to_amm(est, task.amm)
#task.amm.codes=est.A_enc
#task.amm.luts = est.luts.reshape(est.luts.shape[0],-1)
task.mithral_encode_only()
task.lut()

s=timer()
#task.amm.zip_bolt_colmajor_only()
#task.amm.scan_test_zipped()
task.amm.scan_test() #unzipped version
Y_hat=task.amm.out_mat
e=timer()
print("c++ (making own codes & luts, then zipping codes) using sum my debug fn", 1-r2_score(Y, Y_hat))
print(f"Debug version of Mithral Scan faster than Python by: {old_t/(e-s)}")
if r2_score(Y, Y_hat) < 0.99:
  plt.title(f"WARN BASIC COPYING BAD 1-R^2: {1-r2_score(Y, Y_hat):.4f}")
  plt.hist(Y.flatten(),bins=30,label='Y',alpha=0.5)
  plt.hist(Y_hat.flatten(),bins=30,label='Y_hat', alpha=0.5)
  plt.legend()
  plt.show()
else:
  print("C++ debug scan fn is good")

#For Real
task.amm.out_mat=np.zeros(Y.shape)
task.amm.zip_bolt_colmajor_only()
s=timer()
task.scan()
#Y_hat=(task.output()*ncodebooks/task.amm.out_scale) + task.amm.out_offset_sum #use original mithral scan
Y_hat=(task.output()/task.amm.out_scale) + task.amm.out_offset_sum #use original mithral scan
#Y_hat=task.output() #if use my sum
e=timer()
print("Real Scan 1-R^2: ",1-r2_score(Y, Y_hat))
print(f"C++ Wrapped Mithral times faster than Python Matrix Mult: {old_t/(e-s)}")

#%%   
### Tests ###

old_x,old_q=map(np.copy, (X,Q))

def _test_copying_over(iter_X,iter_Q, msg=''):
  y_vs_py_yhat_r2s=[] 
  y_vs_cpp_yhat_r2s=[] 
  cpp_vs_py_r2s=[]
  for X,Q in zip(iter_X, iter_Q):
    Y=X@Q
    est = vq_amm.MithralMatmul(**hparams_dict) #Have to reset to update codes and luts 
    est.fit(X,Q)
    Y_hat_py=est(X,Q)
    task.X=X
    task.Q=Q
    copy_python_to_amm(est, task.amm)
    task.run_matmul(True)
    #Y_hat_cpp=(task.amm.out_mat*ncodebooks/task.amm.out_scale) + task.amm.out_offset_sum
    Y_hat_cpp=(task.amm.out_mat/task.amm.out_scale) + task.amm.out_offset_sum
    
    #task.mithral_encode_only()
    #task.lut()
    ##task.amm.codes=est.A_enc
    ##task.amm.luts = est.luts.reshape(est.luts.shape[0],-1)
    #
    #task.amm.scan_test() #unzipped version
    #Y_hat_cpp=task.amm.out_mat
    print("Py and C++ 1-R^2", 1-r2_score(Y_hat_py, Y_hat_cpp))
    y_vs_py_yhat_r2s+=[1-r2_score(Y, Y_hat_py)]
    y_vs_cpp_yhat_r2s+=[1-r2_score(Y, Y_hat_cpp)]
    cpp_vs_py_r2s+=[1-r2_score(Y_hat_py, Y_hat_cpp)]
  print(msg, y_vs_py_yhat_r2s,y_vs_cpp_yhat_r2s, cpp_vs_py_r2s, sep='\n')
  return list(map(max, (y_vs_py_yhat_r2s,y_vs_cpp_yhat_r2s, cpp_vs_py_r2s)))

def _make_rand_neg(M):
  m=np.copy(M)
  if True: #np.random.rand()  > 0.5:
    every=np.random.randint(1,5)
    m[::every,:]*=-1
    print('every row', every)
  if True: #np.random.rand()  > 0.5:
    every=np.random.randint(1,5)
    m[:,::every]*=-1
    print('every col', every)
  return m

def _make_book_bigger(M):
  m=np.copy(M) 
  c = np.random.randint(ncodebooks)
  #not perfect v Y when >=16
  sz= 2**(np.random.randint(4,7))
  m[:, c*16:(c+1)*16] *= sz 
  m[c*16:(c+1)*16,:] *= sz 
  print('size: ', sz, ' codebook: ', c)
  return m

num_trials=2
results=[]
if False:
  #Compare C++ and Python give similar Y_hat on random Data
  #(They don't)
  print("#####################\n\nrandom floats in 0-1")
  iter_X=(np.random.random_sample(X.shape) for _ in range(num_trials))
  iter_Q=(np.random.random_sample(Q.shape) for _ in range(num_trials))
  results+=[_test_copying_over(iter_X,iter_Q, "Random data so expect to do badly vs Y")]
  
  print("#####################\n\nRandom floats -10 to 10")
  iter_X=(np.random.random_sample(X.shape)*20 -10 for _ in range(num_trials))
  iter_Q=(np.random.random_sample(Q.shape)*20 -10 for _ in range(num_trials))
  results+=[_test_copying_over(iter_X,iter_Q, "Random data so expect to do badly vs Y")]
  
num_trials=5
if True: 
  print("#####################\n\n One codebook a lot bigger")
  iter_X=(X for _ in range(num_trials))
  iter_Q=(_make_book_bigger(Q) for _ in range(num_trials))
  results+=[_test_copying_over(iter_X,iter_Q, "Ordered so do good vs Y")]
  
  print("#####################\n\nOrdered Ints with Some negative")
  iter_X=(_make_rand_neg(X) for _ in range(num_trials))
  iter_Q=(Q for _ in range(num_trials))
  results+=[_test_copying_over(iter_X,iter_Q, "Ordered so do good vs Y")]
  
  
print(list(map(itemgetter(0,1), results)))
print(list(map(itemgetter(2), results)))
#%%
# #Restore
X,Q=old_x,old_q
est = vq_amm.MithralMatmul(**hparams_dict) #Have to reset to update codes and luts 
est.fit(X,Q)
est(X,Q) #to set codes and luts
copy_python_to_amm(est, task.amm)
task.X=X
task.Q=Q
task.scan()

#%%
### Scrap ###
### Scrap ###
# R^2 score of making luts in C++ and Python is >0.999
luts = np.array([np.ravel(est.luts[i], order='C') for i in range(len(est.luts))], dtype=np.uint8)     
print(r2_score(luts, task.amm.luts))


#With random X,Q and copied python params c++ makes the right codes and luts
copy_python_to_amm(est, task.amm)
task.mithral_encode_only()
task.lut()
task.amm.scan_test() #unzipped version
Y_hat=task.amm.out_mat

#Make sure to not zip before this
for i in range(ncodebooks):
  cs=np.sum(task.amm.codes[:,i]!=est.A_enc[:,i]-i*16) 
  assert cs   == 0, cs 
for i in range(ncodebooks*16):
  est_luts=est.luts.reshape(est.luts.shape[0],-1)[:, i]
  ls = np.sum(np.abs(task.amm.luts[:,i]-est_luts) > 1) #off by at most 1
  assert ls   == 0, f"{ls}, {ls/est_luts.size}, {i}" 

plt.hist(Y_hat_py.flatten(),bins=30,label='Y_hat_py', alpha=0.5)
plt.hist(Y_hat.flatten(),bins=30,label='Y_hat',alpha=0.5)
plt.legend()

#%% #Python matches C++ with random codes/luts (C++ overflows?)
task.amm.out_mat =np.zeros(Y.shape)
task.amm.codes=np.random.randint(0, high=16, size=task.amm.codes.size).reshape(task.amm.codes.shape).astype(np.uint8)
task.amm.luts = 50*np.random.randint(1, high=30, size=task.amm.luts.size).reshape(task.amm.luts.shape).astype(np.uint8)
#l = np.zeros(task.amm.luts.shape)
#for i in range(ncodebooks):
#  #l[:,i*16+1]=[1]*(M//2) + [9]*(M//2) #np.random.randint(2, size=M)
#  #l[:,i*16+1]=[i]*(M//2) + [i*4]*(M//2) #np.random.randint(2, size=M)
#  if i %2:
#    l[:,i*16:(i+1)*16] = np.arange(16)
#  else:
#    l[:,i*16:(i+1)*16] = -1*np.arange(15,-1,-1)
#l[:,1]= 40*np.random.randint(4, high=6, size=task.amm.luts.shape[0])
#l[:,0]=[1]*(M//2) + [9]*(M//2) #np.random.randint(2, size=M)
#l[:,17]=np.random.randint(2, size=M)
#task.amm.luts = l
#task.amm.codes = np.array([i  for j in range(N) for i in range(ncodebooks)]).reshape(task.amm.codes.shape)
#task.amm.codes=np.ones(task.amm.codes.shape).astype(np.uint8) 
#task.amm.luts = np.ones(task.amm.luts.shape).astype(np.uint8) *9
task.amm.out_scale=1
task.amm.out_offset_sum=0

est_codes=np.copy(task.amm.codes)
for i in range(ncodebooks):
  est_codes[:,i]+=16*i

#Y_hat_py=dists_enc_cpp(est_codes, task.amm.luts, offset=task.amm.out_offset_sum, scale=task.amm.out_scale, ncodebooks=ncodebooks)
Y_hat_py=dists_enc_short(est_codes, task.amm.luts, offset=task.amm.out_offset_sum, scale=task.amm.out_scale)
#Below doesn't work to adjust for mithral rounding
#Y_hat_py=np.floor((Y_hat_py/ncodebooks)*task.amm.out_scale) + task.amm.out_offset_sum 
task.amm.tmp_codes = task.amm.codes
task.amm.zip_bolt_colmajor_only()
task.scan()
#Round the py version to be accurate comparison
#No codebooks here for comparing against py; scale is already included in codes/luts
Y_hat_cpp=(task.output()/task.amm.out_scale) + task.amm.out_offset_sum 

print("Py and C++ 1-R^2", 1-r2_score(Y_hat_py, Y_hat_cpp))
plt.hist(Y_hat_cpp.flatten(),bins=30,label='Y_hat_cpp',alpha=0.5)
plt.hist(Y_hat_py.flatten(),bins=30,label='Y_hat_py', alpha=0.5)
plt.legend()
plt.show()

print('py unique vals:', len(np.sort(np.unique(Y_hat_py))))
print('C++ unique vals:',len(np.sort(np.unique(Y_hat_cpp))))
#print(np.sort(np.unique(task.amm.luts[:,1] + task.amm.luts[:,17]))/task.amm.out_scale + task.amm.out_offset_sum)


#%% C++ works with fixed codes/luts 
task.amm.out_mat =np.zeros(Y.shape)
task.amm.codes=np.ones(task.amm.codes.shape).astype(float)
task.amm.luts = np.ones(task.amm.luts.shape)*90
task.amm.zip_bolt_colmajor_only()
task.scan()
#Y_hat_cpp=(task.output()*ncodebooks/task.amm.out_scale) + task.amm.out_offset_sum #use original mithral scan
Y_hat_cpp=(task.output()/task.amm.out_scale) + task.amm.out_offset_sum #use original mithral scan
print(Y_hat_cpp)

#%% see what's different
l=est.luts.reshape(est.luts.shape[0],-1)
for i in range(ncodebooks):
  print(i, 'luts', np.sum(l[:,i*16:(i+1)*16]==task.amm.luts[:,i*16:(i+1)*16]))
  print(i, 'codes', np.sum(est.A_enc[:,i]-i*16==task.amm.codes[:,i]))

#task.amm.codes=est.A_enc
#task.amm.luts = est.luts.reshape(est.luts.shape[0],-1)
print('first 64 luts\n', np.ravel( est.luts.reshape(est.luts.shape[0],-1), order='c')[:64])
print('first 32 codes in each column when unzipped (from tmp)')
print(np.ravel( task.amm.tmp_codes, order='f')[:32])
print(np.ravel( task.amm.tmp_codes, order='f')[4096:4096+32])
print(np.ravel( task.amm.tmp_codes, order='f')[2*4096:2*4096+32])
print(np.ravel( task.amm.tmp_codes, order='f')[3*4096:3*4096+32])
print("first 32 codes in each column zipped")
print(np.ravel( task.amm.codes, order='f')[:32])
print(np.ravel( task.amm.codes, order='f')[4096:4096+32])

#%% Runs without segfaulting
Y_hat=est(X,Q)
copy_python_to_amm(est, task.amm)
for i in range(99):
  c, *_ = itemgetter('centroids', 'splitdims','splitvals', 'encode_offsets', 'encode_scales', 'out_offset_sum', 'out_scale')(py_vars)
  task.amm.setCentroidsCopyData(c)
  #task.amm.setCentroids(c)
  task.run_matmul(True)
  print(i)  #always stops at 54?
