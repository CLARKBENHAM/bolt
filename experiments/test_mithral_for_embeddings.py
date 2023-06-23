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
from datetime import datetime

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
#data_sources = [md.load_clip_text_image_tasks()] # slow, didn't load after >30 minutes
text_emb,img_emb = md.load_clip_text_image()
MetricsSoftmax = namedtuple("MetricsSoftmax", ["np_time", "py_fit_time", "py_est_time", "py_est_r2", "py_est_per_ix_kept", "copy_to_cpp_time", "cpp_est_time", "cpp_est_r2", "cpp_est_per_ix_kept"])

# How do you measure the accuracy of embedding search?
# 1. How many of the top-k closest embeddings are in the same class as the query embedding?


#%% # Embedding search
columns=["data_name", "mult_name", "k", "num_queries", "avg_per_same", "latency"]
empty_results = lambda : pd.DataFrame(columns=columns).astype({"data_name": str, "mult_name": str, "k": int, "num_queries": int, "avg_per_same": float})


def compare_on_emb_queries(embeddings, queries,data_name, NREPS,num_queries,ks, ncodebooks=8,lutconsts=-1):
  results = empty_results() 
  

  embeddings_lengths = np.linalg.norm(embeddings, axis=1)
  #embeddings_class=np.apply_along_axis(np.argmax, 1, queries)

  hparams_dict = {'ncodebooks': ncodebooks, 'lut_work_const': lutconsts}
  est = vq_amm.MithralMatmul(**hparams_dict)
  t = time.perf_counter()
  est.fit(embeddings, queries.T)
  py_fit_time=time.perf_counter() - t
  est.predict(embeddings, queries[:100].T) # sets luts, scales, offsets
  
  task=mithral_wrapped.mithral_amm_task_float(*embeddings.shape,num_queries, ncodebooks, lutconsts)
  task.amm.out_mat = np.zeros(task.amm.out_mat.shape)
  task.X=embeddings
  copy_python_to_amm(est, task.amm)
  
  def get_closest_classes(ix, k, closest_embeddings, query_classes):
    closest_classes = np.apply_along_axis(np.argmax, 1, embeddings[closest_embeddings[-k:, ix]]).reshape(-1)
    ct = Counter(closest_classes)
    per_same =100*ct[query_classes[ix]]/k
    return per_same

  def mithral_mult(E,Q):
    #est.set_B(Q)
    #o= est.predict(E,Q)  #this gets much worse after first call (?!!)
    #return o
    
    #copy_python_to_amm(est, task.amm) # Why is this needed?
    task.Q=Q
    
    t = time.perf_counter() 
    task.run_matmul(True)
    Y_hat=task.amm.out_mat #Since we just care about relative order for predicting output
    Y_hat=(Y_hat.astype(np.uint16)*task.amm.ncodebooks/task.amm.out_scale) + task.amm.out_offset_sum
    latency=time.perf_counter() - t
    #Y_hat=(Y_hat.astype(np.float64)*task.amm.ncodebooks/task.amm.out_scale) + task.amm.out_offset_sum
    #print(f"1-r2 {1-r2_score(o, Y_hat)}")
    return Y_hat,latency#[:len()]

  for mult_method, mult_name in ((np.dot, 'numpy'), (mithral_mult, 'mithral')):
  #for mult_method, mult_name in ((mithral_mult, 'mithral'),):
    avg_per_by_k=defaultdict(list)
    for _ in range(NREPS):
      query_ix=np.random.choice(queries.shape[0], num_queries, replace=True)
      search = queries[query_ix, :]
      search_classes = np.apply_along_axis(np.argmax, 1, search).reshape(-1)
      
      if mult_name == 'numpy':
        t = time.perf_counter() 
        mult = np.dot(embeddings,search.T)
        latency=time.perf_counter() - t
      else: 
        mult, latency=mult_method(embeddings,search.T)
      closest_embeddings = np.argsort(
                            np.apply_along_axis(lambda col: col/embeddings_lengths, 
                                                0, 
                                                mult),
                            axis=0)

      for k in ks:
        avg_per_same = sum(map(lambda ix: get_closest_classes(ix, k, closest_embeddings, search_classes), range(num_queries)))/num_queries
        if k == ks[-1]:
          print(f"Avg Percent of top-{k} in same class as query {avg_per_same:.2f}% for {data_name} and {num_queries} queries")
        avg_per_by_k[k].append(avg_per_same)
        new = pd.DataFrame([[data_name, mult_name, k, num_queries, avg_per_same, latency]], columns=columns)
        results=pd.concat([results,new], ignore_index=True)
        
    rep_of_avgs = {k:f"{sum(avgs)/len(avgs):.2f} SD {np.std(avgs):.2f}" for k, avgs in avg_per_by_k.items()}
    print(f"For {data_name} computed by {mult_name} Over {NREPS} of {num_queries} queries by k")
    pprint(rep_of_avgs)

  results['data_name'] = results.data_name.astype('category')
  results['mult_name'] = results.mult_name.astype('category')
  return results

def summary_plot(results, ncodebooks, name=""):
  now=datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
  _dir = os.path.dirname(os.path.abspath(__file__))
  acc_path=os.path.join(_dir, '..', 'experiments', 'results', 'embeddings', f'accuaracy_{name}_{now}.png')
  lat_path=os.path.join(_dir, '..', 'experiments', 'results', 'embeddings', f'latency_{name}_{now}.png')
  
  g = sns.catplot(data = results, y='avg_per_same', x="mult_name", hue= "data_name",col='k', aspect=0.5, kind='swarm')
  g.fig.subplots_adjust(top=0.9) 
  g.fig.suptitle(f"Percent of top-k closest embeddings in same class as query {name}")
  g.fig.savefig(acc_path) 
  print(acc_path)
  
  # no need to seperate by k, all latencies the same
  g = sns.catplot(data = results.query("k==1"), y='latency', x="mult_name", hue= "data_name", aspect=0.5, kind='swarm')
  g.fig.subplots_adjust(top=0.9) 
  g.fig.suptitle(f"Latency of computing {num_queries} queries with {ncodebooks} {name}")
  g.fig.savefig(lat_path) 
  print(lat_path)
  
num_queries=32 #512*8
NREPS = 50
ncodebooks=16
if True: #hardcode
  ks=[1,5,10,100,1000]
  name='clip_text_q_img_ix'
  text_emb_subset = text_emb[np.random.choice(text_emb.shape[0], 10000, replace=True)]
  results=compare_on_emb_queries(img_emb, text_emb_subset,name, NREPS, num_queries,ks,ncodebooks=8)
  summary_plot(results, ncodebooks, name=name)
else:
  ks=[1,5,10,100]
  results = empty_results() 
  for data in itertools.chain(*data_sources[1:]):
    [W_test,W_train, X_test, X_train, Y_test, Y_train] = attrgetter('W_test','W_train', 'X_test', 'X_train', 'Y_test', 'Y_train')(data)
    
    #ncodebooks=2**math.floor(np.log2(Y_train.shape[1])) #>16 errors on cifar100
    queries = Y_test[:-(len(Y_test)%32)]
    embeddings = Y_train[:-(len(Y_train)%32)]
    new=compare_on_emb_queries(embeddings, queries, data.name, NREPS, num_queries,ks,ncodebooks=ncodebooks)
    results = pd.concat([new, results],ignore_index=True)
  summary_plot(results, ncodebooks, name=data.name)
#%% ###     Scrap
k=100
closest_embeddings = np.argsort((embeddings @ query.T).reshape(-1) / embeddings_lengths)
ct= defaultdict(int)
for ix,a in enumerate(embeddings[closest_embeddings[-k:]]):
  cosine_sim=np.dot(a, query.reshape(-1))/ (np.linalg.norm(a) * np.linalg.norm(query))
  embedding_class =np.argmax(a)
  print(cosine_sim, embedding_class, search_classes)
  ct[embedding_class] += 1
print(f"Percent of top-{k} in same class as query {100*ct[search_classes[0]]/k:.2f}%")

#%%
t = time.perf_counter()

lutconsts=-1
np_time=time.perf_counter() - t