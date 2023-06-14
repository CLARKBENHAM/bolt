#%% A pretty version of mithral speed tests, with graphs
# Meant to be run as notebook; run from bolt/experiments directory if as script

import re
import os
import sys
import io
import math
from contextlib import contextmanager
import ctypes
import tempfile
import time
from collections import namedtuple
from timeit import default_timer as timer
import numpy as np
from sklearn.metrics import r2_score
from operator import attrgetter, itemgetter
import matplotlib.pyplot as plt
from matplotlib import colors
import seaborn as sns
import itertools

from python import matmul_datasets as md 
from python import vq_amm
from copy_to_amm import copy_python_to_amm, extract_py_vars, copy_python_luts

try:
  repo_path=os.path.join(os.path.dirname(__file__), "..")
except:
  #added except since when vscode auto-restarts after crashes, __file__ isn't defined
  repo_path = os.path.abspath('')[:os.path.abspath('').index('/bolt/') + 6]
sys.path.append(repo_path)
from cpp import mithral_wrapped

import functools
print = functools.partial(print, flush=True)

assert 5 == mithral_wrapped.add(2, 3) #imports worked


data_sources = [md.load_cifar10_tasks(), md.load_cifar100_tasks()]
#%%
MetricsSoftmax = namedtuple("MetricsSoftmax", ["np_time", "py_fit_time", "py_est_time", "py_est_r2", "py_est_per_ix_kept", "copy_to_cpp_time", "cpp_est_time", "cpp_est_r2", "cpp_est_per_ix_kept"])

#Run on last layers of NN
results=[]
results_std=[]
ncodebooks=2
NREPS=10
NAVG=5
print(f"ncodebooks={ncodebooks}")
for data in itertools.chain(*data_sources):
  print("$$$$$data", data.name)
  min_trials = []
  for _ in range(NAVG):
    trials=[]
    for _ in range(NREPS):
      [W_test,W_train, X_test, X_train, Y_test, Y_train] = attrgetter('W_test','W_train', 'X_test', 'X_train', 'Y_test', 'Y_train')(data)
      #Mithral C++ doesn't work with counts not aligned to 32
      align32=len(Y_test)-(len(Y_test)%32)
      Y_test=Y_test[:align32]
      X_test=X_test[:align32]
      lutconsts=-1
      t = time.perf_counter()
      Y=X_test@W_test
      np_time=time.perf_counter() - t
      #mse=np.mean((np.abs(np.ravel(Y) - np.ravel(Y_test)))**2)
      #assert mse < 0.001*Y.size, mse
      max_ix=np.apply_along_axis(np.argmax, 1, Y_test)
      
      hparams_dict = {'ncodebooks': ncodebooks, 'lut_work_const': lutconsts}
      est = vq_amm.MithralMatmul(**hparams_dict)
      t = time.perf_counter()
      est.fit(X_train,W_train)
      py_fit_time=time.perf_counter() - t
 
      t = time.perf_counter()
      Y_hat1 = est.predict(X_test, W_test)
      py_est_time=time.perf_counter() - t
      py_est_r2 = r2_score(Y,Y_hat1)
      py_max_ix=np.apply_along_axis(np.argmax, 1, Y_hat1)
      py_est_per_ix_kept=np.sum(py_max_ix==max_ix)/py_max_ix.size
      
      task=mithral_wrapped.mithral_amm_task_float(*X_test.shape,W_test.shape[1], ncodebooks, lutconsts)
      task.amm.out_mat = np.zeros(task.amm.out_mat.shape)
      t = time.perf_counter()
      task.X=X_test
      task.Q=W_test
      copy_python_to_amm(est, task.amm)
      copy_python_luts(est, task.amm) # Or can make luts in C++
      copy_to_cpp_time=time.perf_counter() - t

      #task.X=X_train[:len(X_test)]
      #task.lut()#use X known at train time
      #task.X =X_test
      t = time.perf_counter()
      task.run_matmul(False)
      #task.run_matmul(True) #Encodes test X as centroids instead of using train_x's centroids
      Y_hat2=task.amm.out_mat #Since we just care about relative order for predicting output
      cpp_est_time=time.perf_counter() - t
      Y_hat2=(Y_hat2.astype(np.uint16)*ncodebooks/task.amm.out_scale) + task.amm.out_offset_sum
      cpp_est_r2=r2_score(Y, Y_hat2)
      cpp_max_ix=np.apply_along_axis(np.argmax, 1, Y_hat2)
      cpp_est_per_ix_kept=np.sum(cpp_max_ix==max_ix)/cpp_max_ix.size
      o= MetricsSoftmax(np_time, py_fit_time, py_est_time, py_est_r2, py_est_per_ix_kept, copy_to_cpp_time, cpp_est_time, cpp_est_r2, cpp_est_per_ix_kept)
      
      trials += [o]
    min_trials += [MetricsSoftmax(*np.min(trials, axis=0))]
  print(f"##Each Min Trial of {NREPS} for {data.name}##")
  attr=['np_time', 'py_est_r2', 'py_est_per_ix_kept', 'cpp_est_time', 'cpp_est_r2', 'cpp_est_per_ix_kept']
  print(attr)
  for o in min_trials:
    print(attrgetter(*attr)(o))
    
  results += [MetricsSoftmax(*np.average(min_trials, axis=0))]
  results_std += [MetricsSoftmax(*np.std(min_trials, axis=0))]
  
min_np_times=list(map(lambda i: i.np_time, results))
min_mithral_times=list(map(lambda i: i.cpp_est_time, results))
cpp_est_per_ix_kepts=list(map(lambda i: i.cpp_est_per_ix_kept, results))
print(f"ncodebooks={ncodebooks}")
print(f"Avg over {NAVG} of Min of {NREPS} Numpy Matrix mult times: {min_np_times}, {[i.np_time for i in results_std]}")
print(f"Avg over {NAVG} of Min of {NREPS} Mithral times: {min_mithral_times}, {[i.cpp_est_time for i in results_std]}")
print(f"Avg over {NAVG} of Min of {NREPS} cpp_est_per_ix_keps: {cpp_est_per_ix_kepts}")
print(f"Py/C++ time ratio: {[i/j for i,j in zip(min_np_times, min_mithral_times)]}")

#%%
plt.title("Raw Y")
plt.hist(Y_test.flatten(),bins=30,label='Y',alpha=0.3)
plt.hist(Y_hat1.flatten(),bins=30,label='Y_hat1 (py)', alpha=0.3)
plt.hist(Y_hat2.flatten(),bins=30,label='Y_hat2 (cpp)', alpha=0.3)
plt.legend()
plt.show()

plt.title("Max Ix")
plt.hist(max_ix,label='Y_ix',alpha=0.3)
plt.hist(py_max_ix,label='Y_hat1_ix (py)', alpha=0.3)
plt.hist(cpp_max_ix,label='Y_hat2_ix (cpp)', alpha=0.3)
plt.legend()
plt.show()

#%% Make Comparison Graphs 
Metrics = namedtuple("Metrics", ["np_time", "py_fit_time", "py_est_time", "py_est_r2", "copy_to_cpp_time", "cpp_est_time", "cpp_est_r2"])

def compute_metrics_no_train(N,D,M,ncodebooks,X,Q):
  lutconsts=-1
  task=mithral_wrapped.mithral_amm_task_float(N,D,M, ncodebooks, lutconsts)
  #ignoring the time to copy to c++ initally

  s=timer()
  Y=X@Q
  e=timer()
  np_time=e-s #in cpu sec's
 
  hparams_dict = {'ncodebooks': ncodebooks, 'lut_work_const': lutconsts}
  est = vq_amm.MithralMatmul(**hparams_dict)
  s=timer()
  est.fit(X,Q)
  e=timer()
  py_fit_time=e-s
 
  s=timer()
  Y_hat1=est(X, Q)  #this will be wrong; still using preprocess_x_like_cpp
  e=timer()
  py_est_time=e-s
  py_est_r2 = r2_score(Y_hat1,Y)
  
  s=timer()
  task.X=X
  task.Q=Q
  copy_python_to_amm(est, task.amm)
  e=timer()
  copy_to_cpp_time=e-s	
 
  s=timer()
  task.run_matmul(True)
  Y_hat=(task.amm.out_mat*ncodebooks/task.amm.out_scale) + task.amm.out_offset_sum
  #multiply by ncodebooks since out_mat is average of correct values
  e=timer()
  cpp_est_time=e-s
  cpp_est_r2=r2_score(Y, Y_hat)
  return Metrics(np_time, py_fit_time, py_est_time, py_est_r2, copy_to_cpp_time, cpp_est_time, cpp_est_r2)

def extract_attr_from_metrics(mat, attr):
  "extract attr from tensor of Metrics tuples"
  ix=Metrics._fields.index(attr)
  return np.fromiter((o[ix] for o in np.ravel(mat)), float).reshape(mat.shape)

def make_heatmap(x,y,vectorized_fn, Z1=None):
  """  
  Z1=vectorized_fn on a gridmap made from x and y if Z1 is undefined.
  Start on times and R^2 heatmap from Z1
  Returns:
      array:  [fig, (ax1,ax2), Z1]
  """
  x_grid,y_grid = np.meshgrid(x,y)
  if Z1 is None:
    Z1=vectorized_fn(x_grid,y_grid)
  
  times = extract_attr_from_metrics(Z1,'np_time') / extract_attr_from_metrics(Z1, 'cpp_est_time') 
  r2 = extract_attr_from_metrics(Z1,'cpp_est_r2')
  xticklabels = list(map(str, x))
  yticklabels = list(map(str, y))
  
  fig, (ax1, ax2) = plt.subplots(1, 2)
  sns.heatmap(times, ax=ax1, cbar=True,  norm=colors.LogNorm(),vmin=0.1, vmax=100,
              xticklabels=xticklabels, yticklabels=yticklabels)
  ax1.set_title('Times Faster')
  
  sns.heatmap(r2, ax=ax2, cbar=True, vmin=0, vmax=1,
              xticklabels=xticklabels, yticklabels=yticklabels)
  ax2.set_title('R^2')
 
  return fig, (ax1, ax2), Z1
 
vectorize_ret_metrics = functools.partial(np.vectorize, otypes=[Metrics])

#%%
@vectorize_ret_metrics
def compute_metrics_no_train_NM_simple(N,M):
  D=int(N/64)
  ncodebooks=2
  X= np.stack([np.array([i%16]*(D//2) + [(i%16) for j in range(D//2)]) for i in range(N)])
  Q= np.stack([np.array([i%16]*(M//2) + [16 + i%16]*(M//2)) for i in range(D)])
  return compute_metrics_no_train(N,D,M,ncodebooks,X,Q)

#Ns = np.array([4096, 32768, 262144 ]).astype(int) #Kernel Times out
Ns = np.array([4096, 16384 ]).astype(int)
Ms = np.array([128,512,2048,8192]).astype(int)

fig, axs,Z1 = make_heatmap(Ns, Ms, compute_metrics_no_train_NM_simple)
fig.suptitle("Basic X and Q, ordered ints from 0-15")
for ax in axs:
  ax.set_xlabel('N dim sz, D sz=N/64')
  ax.set_ylabel('M dim sz')
plt.savefig('sequential_0_16_XQ_changing_DM.png')
plt.show()

print(Z1)

#%%
@vectorize_ret_metrics
def compute_metrics_no_train_NM_random(N,M):
  D=int(N/64)
  ncodebooks=2
  #X= np.random.uniform(low=0, high=10, size=(N,D))
  #Q= np.random.normal(size=(D,M))*10 #will this work with negative numbers?
  X= np.stack([np.array([i%16]*(D//2) + [(i%16) for j in range(D//2)]) for i in range(N)])
  Q= np.stack([np.array([i%16]*(M//2) + [16 + i%16]*(M//2)) for i in range(D)])
  return compute_metrics_no_train(N,D,M,ncodebooks,X,Q)

#Ns = np.array([4096, 32768, 262144 ]).astype(int) #Kernel Times out
Ns = np.array([4096, 16384 ]).astype(int)
Ms = np.array([128,512,2048,8192]).astype(int)

fig, axs,Z3 = make_heatmap(Ns, Ms, compute_metrics_no_train_NM_random)
fig.suptitle("Basic X and Q, ordered ints from 0-15")
for ax in axs:
  ax.set_xlabel('N dim sz, D sz=N/64')
  ax.set_ylabel('M dim sz')
plt.savefig('random_XQ_changing_DM.png')
plt.show()

print(Z3)

#%%  
@vectorize_ret_metrics
def compute_metrics_no_train_DNC_random(D, ncodebooks):
  """Random data change ncodebooks and size of D"""
  N=8192
  M=256
  X= np.random.uniform(low=1, high=100, size=(N,D))
  Q= np.random.normal(size=(D,M))*10 #will this work with negative numbers?
  return compute_metrics_no_train(N,D,M,ncodebooks,X,Q)

Ds= np.array([64,128,256])
Ncodebooks=np.array([2,4,8,16,32])

fig, axs,Z2 = make_heatmap(Ds, Ncodebooks, compute_metrics_no_train_DNC_random)
fig.suptitle("X in U[0,10] and Q in N(0,1)*10, N=8192,M=256")
for ax in axs:
  ax.set_xlabel('D dim sz')
  ax.set_ylabel('Ncodebooks')
plt.savefig('random_XQ_changing_codebooks_D.png')
plt.show()

print(Z2)



#%% Scrap
N,D,M = 4096, 64,128  
ncodebooks=2
X= np.stack([np.array([i%16]*(D//2) + [(i%16) for j in range(D//2)]) for i in range(N)])
Q= np.stack([np.array([i%16]*(M//2) + [16 + i%16]*(M//2)) for i in range(D)])
o=compute_metrics_no_train(N,D,M,ncodebooks,X,Q)
print(compute_metrics_no_train(N,D,M,ncodebooks,X,Q))


#%% See how good it is if it knows

task=mithral_wrapped.mithral_amm_task_float(*X_train.shape,W_train.shape[1], ncodebooks, lutconsts)
task.amm.out_mat = np.zeros(task.amm.out_mat.shape)
t = time.perf_counter()
task.X=X_train
task.Q=W_train
copy_python_to_amm(est, task.amm)
task.run_matmul(True)
Y_hat2=(task.amm.out_mat[:Y.shape[0]]*ncodebooks/task.amm.out_scale) + task.amm.out_offset_sum
cpp_est_r2=r2_score(Y, Y_hat2)
cpp_max_ix=np.apply_along_axis(np.argmax, 1, Y_hat2)
cpp_est_per_ix_kept=np.sum(cpp_max_ix==max_ix)/cpp_max_ix.size
print(cpp_est_r2, cpp_est_per_ix_kept)
















