#%% A pretty version of mithral speed tests, with graphs

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
from matplotlib import colors
import seaborn as sns

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

assert 5 == mithral_wrapped.add(2, 3) #imports worked


def _reshape_split_lists(enc, att: str):
  #(ncodebooks x 4) for values for ncodebook subspaces to pick from the 4 levels 
  #aka ncodebooks by the 4 dims to split on for each output
  return np.array([
    getattr(i, att)
    for a in enc.splits_lists 
    for i in a]).reshape(enc.ncodebooks,4)

def extract_py_vars(est):
  """Munges and casts python param data to work for C++"""
  
  #py est splitvals is jagged 3d (ncodebooks x 4 x [1,2,4,8]). Reshape to (ncodebooks x 16)
  raw_splitvals=[[v for i in a for v in i.vals]
                     for a in est.enc.splits_lists 
                     ]
  default_sz = max(map(len, raw_splitvals))
  ## i idx is 2^{i-1} value can split nodes for all the ncodebook subspaces
  #C++ expects 0 padded values out to 16 per BST of split values
  #in C++ uses (ncodebooks, 32) if row major accesses
  # iterate over the 32 for a given codebook
  # [v1,v2,v2,v3,v3,v3,v3,v4,v4,v4,v4,v4,v4,v4,v4]
  # (nsplits, [1,2,4,8]) 
  #but insert 0 to pad to 16 with 1 indexed array
  
  raw_splitvals_padded=np.array([np.pad(l, (1,0))
            for l in raw_splitvals
            ])
  #Python: (X-offset) * scale; C++: X*scale + offset; before comparing to these splitvals
  #WARN: these can overflow sometimes from int8 
  splitvals=(raw_splitvals_padded-128).clip(-128,127).astype(np.int8)
  
  encode_scales = _reshape_split_lists(est.enc, 'scaleby').astype(np.float32)
  raw_encode_offset = _reshape_split_lists(est.enc, 'offset').astype(np.float32) 
  c = est.enc.centroids.astype(np.float32)
  reshaped_centroids=np.concatenate(list(map(lambda cbook_ix : np.ravel(c[cbook_ix], order='f'), range(len(c)))))
  return {
      "splitdims": _reshape_split_lists(est.enc, 'dim').astype(np.uint32), #these are what's called idxs in paper?
      "splitvals": splitvals,
      "encode_scales": encode_scales, 
      "encode_offsets": raw_encode_offset*-encode_scales - 128, 
      "centroids": reshaped_centroids,
      #"idxs": ,  #only need to set if we have a sparse matrix; idxs is used in mithral_lut_sparse always, but idxs are fine by default
      # if lut_work_const then c++ is row (ncodebooks,D) set to ncodebooks*[range(D)], else its ncodebooks*[sorted(permutation(range(D), nnz_per_centroid))]
      #Python encode_X (idxs?!) = offsets into raveled indxs + idxs=[[i%16]*ncodebooks for i in range(N)]), the offset is added to each idxs row
      #cpp out_offset_sum/out_scale is set by results at _compute_offsets_scale_from_mins_maxs, after done learning luts in mithral_lut_dense&mithral_lut_spares. no need to copy
      #   But then it's never used?
      "out_offset_sum": np.float32(est.offset),
      "out_scale":      np.float32(est.scale),
  }

def copy_python_to_amm(py_est, amm):
  py_vars = extract_py_vars(py_est)
  [c, d,v,eo,es, osum, oscale] = itemgetter('centroids', 'splitdims','splitvals', 'encode_offsets', 'encode_scales', 'out_offset_sum', 'out_scale')(py_vars)

  amm.setCentroidsCopyData(c)
  amm.setSplitdims(d)
  amm.setSplitvals(v)
  amm.setEncode_scales(es) 
  amm.setEncode_offsets(eo)
  amm.out_offset_sum = osum
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

Metrics = namedtuple("Metrics", ["np_time", "py_fit_time", "py_est_time", "py_est_r2", "copy_to_cpp_time", "cpp_est_time", "cpp_est_r2"])

def compute_metrics(N,D,M,ncodebooks,X,Q):
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
def compute_metrics_NM_simple(N,M):
  D=int(N/64)
  ncodebooks=2
  X= np.stack([np.array([i%16]*(D//2) + [(i%16) for j in range(D//2)]) for i in range(N)])
  Q= np.stack([np.array([i%16]*(M//2) + [16 + i%16]*(M//2)) for i in range(D)])
  return compute_metrics(N,D,M,ncodebooks,X,Q)

#Ns = np.array([4096, 32768, 262144 ]).astype(int) #Kernel Times out
Ns = np.array([4096, 16384 ]).astype(int)
Ms = np.array([128,512,2048,8192]).astype(int)

fig, axs,Z1 = make_heatmap(Ns, Ms, compute_metrics_NM_simple)
fig.suptitle("Basic X and Q, ordered ints from 0-15")
for ax in axs:
  ax.set_xlabel('N dim sz, D sz=N/64')
  ax.set_ylabel('M dim sz')
plt.savefig('sequential_0_16_XQ_changing_DM.png')
plt.show()

print(Z1)

#%%
@vectorize_ret_metrics
def compute_metrics_NM_random(N,M):
  D=int(N/64)
  ncodebooks=2
  #X= np.random.uniform(low=0, high=10, size=(N,D))
  #Q= np.random.normal(size=(D,M))*10 #will this work with negative numbers?
  X= np.stack([np.array([i%16]*(D//2) + [(i%16) for j in range(D//2)]) for i in range(N)])
  Q= np.stack([np.array([i%16]*(M//2) + [16 + i%16]*(M//2)) for i in range(D)])
  return compute_metrics(N,D,M,ncodebooks,X,Q)

#Ns = np.array([4096, 32768, 262144 ]).astype(int) #Kernel Times out
Ns = np.array([4096, 16384 ]).astype(int)
Ms = np.array([128,512,2048,8192]).astype(int)

fig, axs,Z3 = make_heatmap(Ns, Ms, compute_metrics_NM_random)
fig.suptitle("Basic X and Q, ordered ints from 0-15")
for ax in axs:
  ax.set_xlabel('N dim sz, D sz=N/64')
  ax.set_ylabel('M dim sz')
plt.savefig('random_XQ_changing_DM.png')
plt.show()

print(Z3)

#%%  
@vectorize_ret_metrics
def compute_metrics_DNC_random(D, ncodebooks):
  """Random data change ncodebooks and size of D"""
  N=8192
  M=256
  X= np.random.uniform(low=1, high=100, size=(N,D))
  Q= np.random.normal(size=(D,M))*10 #will this work with negative numbers?
  return compute_metrics(N,D,M,ncodebooks,X,Q)

Ds= np.array([64,128,256])
Ncodebooks=np.array([2,4,8,16,32])

fig, axs,Z2 = make_heatmap(Ds, Ncodebooks, compute_metrics_DNC_random)
fig.suptitle("X in U[0,10] and Q in N(0,1)*10, N=8192,M=256")
for ax in axs:
  ax.set_xlabel('D dim sz')
  ax.set_ylabel('Ncodebooks')
plt.savefig('random_XQ_changing_codebooks_D.png')
plt.show()

print(Z2)



#%%
N,D,M = 4096, 64,128 
ncodebooks=2
X= np.stack([np.array([i%16]*(D//2) + [(i%16) for j in range(D//2)]) for i in range(N)])
Q= np.stack([np.array([i%16]*(M//2) + [16 + i%16]*(M//2)) for i in range(D)])
o=compute_metrics(N,D,M,ncodebooks,X,Q)
print(compute_metrics(N,D,M,ncodebooks,X,Q))
















