#%%

# Fiting on the data that's about to be muliplied still only gives an accuracy of 
import re
import os
import sys
import io
import math
from contextlib import contextmanager
import ctypes
import tempfile
import time
from collections import namedtuple, defaultdict, Counter
from timeit import default_timer as timer
import numpy as np
from sklearn.metrics import r2_score
from operator import attrgetter, itemgetter
import matplotlib.pyplot as plt
from matplotlib import colors
import seaborn as sns
import itertools
from pprint import pprint 
import pandas as pd 
import seaborn as sns 

from python import matmul_datasets as md 
from python import vq_amm
from copy_to_amm import extract_py_vars, copy_python_luts, copy_python_to_amm

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
MetricsSoftmax = namedtuple("MetricsSoftmax", ["np_time", "py_fit_time", "py_est_time", "py_est_r2", "py_est_per_ix_kept", "copy_to_cpp_time", "cpp_est_time", "cpp_est_r2", "cpp_est_per_ix_kept"])

# How do you measure the accuracy of embedding search?
# 1. How many of the top-k closest embeddings are in the same class as the query embedding?

def get_closest_classes(ix, k, closest_embeddings, query_classes):
  closest_classes = np.apply_along_axis(np.argmax, 1, embeddings[closest_embeddings[-k:, ix]]).reshape(-1)
  ct = Counter(closest_classes)
  per_same =100*ct[query_classes[ix]]/k
  #print(f"Percent of top-{k} in same class {query_classes[ix]} as query {per_same:.2f}%")
  return per_same


#%% # Embedding search
columns=["data_name", "mult_name", "k", "num_queries", "avg_per_same", "latency"]
results = pd.DataFrame(columns=columns).astype({"data_name": str, "mult_name": str, "k": int, "num_queries": int, "avg_per_same": float})

lutconsts=-1
ncodebooks=16
num_queries=64 #512*8
#assert(num_queries%32==0)
NREPS = 50
#ks=[1,2,5,10,20,50,100]
ks=[1,5,10,100]
for data in itertools.chain(*data_sources[1:]):
  [W_test,W_train, X_test, X_train, Y_test, Y_train] = attrgetter('W_test','W_train', 'X_test', 'X_train', 'Y_test', 'Y_train')(data)
  #Y_test = Y_train
  #Y_test = np.ones(Y_train.shape)
  Y_test = Y_test[:-(len(Y_test)%32)]
  Y_train = Y_train[:-(len(Y_train)%32)]
  embeddings = Y_train
  embeddings_lengths = np.linalg.norm(embeddings, axis=1)
  embeddings_class=np.apply_along_axis(np.argmax, 1, Y_test)

  hparams_dict = {'ncodebooks': ncodebooks, 'lut_work_const': lutconsts}
  est = vq_amm.MithralMatmul(**hparams_dict)
  t = time.perf_counter()
  est.fit(Y_train, Y_test.T)
  py_fit_time=time.perf_counter() - t
  est.predict(Y_train, Y_test[:100].T) # sets luts, scales, offsets
  
  task=mithral_wrapped.mithral_amm_task_float(*Y_train.shape,num_queries, ncodebooks, lutconsts)
  task.amm.out_mat = np.zeros(task.amm.out_mat.shape)
  task.X=embeddings
  #print('before copy')
  #print(task.amm.getEncode_scales())
  #print(task.amm.getEncode_offsets())
  copy_python_to_amm(est, task.amm)
  #copy_python_luts(est, task.amm) # Or can make luts in C++
  #print('after py copy')
  #print(task.amm.getEncode_scales())
  #print(task.amm.getEncode_offsets())
  py_vars = extract_py_vars(est)
  [c, d,v,eo,es, osum, oscale] = itemgetter('centroids', 'splitdims','splitvals', 'encode_offsets', 'encode_scales', 'out_offset_sum', 'out_scale')(py_vars)
  ## Neither scales nor offsets stay changed after copy. They're correct inside fn at end, but revert back here
  assert np.all(np.ravel(task.amm.getCentroids()) == np.ravel(c)) 
  assert np.all(task.amm.getSplitdims() == d)
  #
  # Both asserts fail, but had to run in copy_python_to_amm!
  assert np.all(task.amm.getEncode_scales()==es)
  assert np.all(task.amm.getEncode_offsets() == eo)
  #
  ### Why do these need to change for each Q? These affect X
  ### Something about task is getting modifed when I run it twice?
  ##task.amm.setSplitdims(d)
  #print('after task set copy')
  #task.amm.setEncode_scales(es) 
  #task.amm.setEncode_offsets(eo)
  #print(task.amm.getEncode_scales())
  #print(task.amm.getEncode_offsets())
  ##print(task.amm.getSplitdims())
  ##print(task.amm.getSplitvals())

  #task.amm.setSplitvals(v)
  
  def mithral_mult(E,Q):
    #est.set_B(Q)
    #o= est.predict(E,Q)  #this gets much worse after first call (?!!)
    #return o
    
    #copy_python_to_amm(est, task.amm) # Why is this needed?
    task.Q=Q
    
    task.run_matmul(True)
    Y_hat=task.amm.out_mat #Since we just care about relative order for predicting output
    Y_hat=(Y_hat.astype(np.uint16)*task.amm.ncodebooks/task.amm.out_scale) + task.amm.out_offset_sum
    #Y_hat=(Y_hat.astype(np.float64)*task.amm.ncodebooks/task.amm.out_scale) + task.amm.out_offset_sum
    #print(f"1-r2 {1-r2_score(o, Y_hat)}")
    return Y_hat#[:len()]

  for mult_method, mult_name in ((np.dot, 'numpy'), (mithral_mult, 'mithral')):
  #for mult_method, mult_name in ((mithral_mult, 'mithral'),):
    avg_per_by_k=defaultdict(list)
    for _ in range(NREPS):
      query_ix=np.random.choice(Y_test.shape[0], num_queries, replace=True)
      queries = Y_test[query_ix, :]
      query_classes = np.apply_along_axis(np.argmax, 1, queries).reshape(-1)
     
      t = time.perf_counter() 
      closest_embeddings = np.argsort(
                            np.apply_along_axis(lambda col: col/embeddings_lengths, 
                                                0, 
                                                mult_method(embeddings,queries.T)),
                            axis=0)
      latency=time.perf_counter() - t

      for k in ks:
        avg_per_same = sum(map(lambda ix: get_closest_classes(ix, k, closest_embeddings, query_classes), range(num_queries)))/num_queries
        if k == ks[-1]:
          print(f"Avg Percent of top-{k} in same class as query {avg_per_same:.2f}% for {data.name} and {num_queries} queries")
        avg_per_by_k[k].append(avg_per_same)
        new = pd.DataFrame([[data.name, mult_name, k, num_queries, avg_per_same, latency]], columns=columns)
        results=pd.concat([results,new], ignore_index=True)
        
    rep_of_avgs = {k:f"{sum(avgs)/len(avgs):.2f} SD {np.std(avgs):.2f}" for k, avgs in avg_per_by_k.items()}
    print(f"For {data.name} computed by {mult_name} Over {NREPS} of {num_queries} queries by k")
    pprint(rep_of_avgs)

results['data_name'] = results.data_name.astype('category')
results['mult_name'] = results.mult_name.astype('category')
# print(results)
#%
#sns.swarmplot(data = results, y='avg_per_same', x='k')

g = sns.catplot(data = results, y='avg_per_same', x="mult_name", hue= "data_name",col='k', aspect=0.5, kind='swarm')
g.fig.subplots_adjust(top=0.9) 
g.fig.suptitle("Percent of top-k closest embeddings in same class as query")
#%%
# no need to seperate by k, all latencies the same
g = sns.catplot(data = results.query("k==1"), y='latency', x="mult_name", hue= "data_name", aspect=0.5, kind='swarm')
g.fig.subplots_adjust(top=0.9) 
g.fig.suptitle(f"Latency of computing {num_queries} queries with {ncodebooks}")

#%% ###     Scrap
k=100
closest_embeddings = np.argsort((embeddings @ query.T).reshape(-1) / embeddings_lengths)
ct= defaultdict(int)
for ix,a in enumerate(embeddings[closest_embeddings[-k:]]):
  cosine_sim=np.dot(a, query.reshape(-1))/ (np.linalg.norm(a) * np.linalg.norm(query))
  embedding_class =np.argmax(a)
  print(cosine_sim, embedding_class, query_classes)
  ct[embedding_class] += 1
print(f"Percent of top-{k} in same class as query {100*ct[query_classes[0]]/k:.2f}%")

#%%
t = time.perf_counter()

lutconsts=-1
np_time=time.perf_counter() - t