#%% Can PyTorch serialize a pybind11 function?

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
import torch

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
data = data_sources[0][0]
[W_test,W_train, X_test, X_train, Y_test, Y_train] = attrgetter('W_test','W_train', 'X_test', 'X_train', 'Y_test', 'Y_train')(data)
#Mithral C++ doesn't work with counts not aligned to 32
align32=len(X_test)-(len(X_test)%32)
X_test=X_test[:align32]
Y_test=Y_test[:align32]
#%%
def make_task(X_train, W_train, ncodebooks = 4):
  lutconsts=-1
  hparams_dict = {'ncodebooks': ncodebooks, 'lut_work_const': lutconsts}
  #est = vq_amm.MithralMatmul(**hparams_dict)
  est = vq_amm.MithralMatmul(ncodebooks=ncodebooks, lut_work_const=lutconsts)
  est.fit(X_train,W_train)
  est.predict(X_test, W_test) # this sets scale, offset, and luts
  #est.scale = 1
  #est.offset = 0
  task=mithral_wrapped.mithral_amm_task_float(*X_test.shape,W_test.shape[1], ncodebooks, lutconsts)
  task.amm.out_mat = np.zeros(task.amm.out_mat.shape)

  copy_python_to_amm(est, task.amm)
  if True:# copy luts 
    copy_python_luts(est, task.amm) 
  return task

task = make_task(X_train, W_train)


def compute(task, X_test, W_test):
  task.X=X_test
  task.Q=W_test
     
  #task.run_matmul(False)
  task.run_matmul(True) #Encodes test Q as LUT instead of using train_Q's luts 
  Y_hat2=task.amm.out_mat #Since we just care about relative order for predicting output
  Y_hat2=(Y_hat2.astype(np.uint16)*task.amm.ncodebooks/task.amm.out_scale) + task.amm.out_offset_sum
  cpp_max_ix=np.apply_along_axis(np.argmax, 1, Y_hat2)
  return cpp_max_ix

cpp_max_ix = compute(task, X_test, W_test)
Y=X_test@W_test
max_ix=np.apply_along_axis(np.argmax, 1, Y)
cpp_est_per_ix_kept=np.sum(cpp_max_ix==max_ix)/cpp_max_ix.size
print("Percent of max_ix kept: ", cpp_est_per_ix_kept*100)

#%%  Can you serialize a pybind11 function?
# errors with attribute lookup is not defined on python value of type 'mithral_amm_task_float':
# task.run_matmul(True)

class MithralNN(torch.nn.Module):
  def __init__(self, X_test, W_test, ncodebooks, lutconsts):
    super().__init__()
    self._W = torch.nn.Parameter(torch.arange(10), requires_grad=False)
    self.task=mithral_wrapped.mithral_amm_task_float(*X_test.shape,W_test.shape[1], ncodebooks, lutconsts)
    
  def forward(self, X):
    #W = self._W.repeat(X.shape[0])
    #task = make_task(X, W)
    task.run_matmul(True) #Encodes test Q as LUT instead of using train_Q's luts 
    Y_hat2=task.amm.out_mat #Since we just care about relative order for predicting output
    return Y_hat2 * np.mean(X)
  
scripted_foo = torch.jit.script(MithralNN(X_test, W_test,4,-1))
print(scripted_foo.graph)
#%%