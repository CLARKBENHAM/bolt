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
from functools import partial
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
from copy_to_amm import extract_py_vars, copy_python_luts, copy_python_to_amm, extract_mithral_vars

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
print(os.getpid())

data_sources = [md.load_cifar10_tasks(), md.load_cifar100_tasks()]
#data_sources = [md.load_clip_text_image_tasks()] # slow, didn't load after >30 minutes
text_emb,img_emb = md.load_clip_text_image()
text_emb = text_emb[:1000]
img_emb = img_emb[:1000]
MetricsSoftmax = namedtuple("MetricsSoftmax", ["np_time", "py_fit_time", "py_est_time", "py_est_r2", "py_est_per_ix_kept", "copy_to_cpp_time", "cpp_est_time", "cpp_est_r2", "cpp_est_per_ix_kept"])

# How do you measure the accuracy of embedding search?
# 1. How many of the top-k closest embeddings are in the same class as the query embedding?


#%% # Embedding search
np.random.seed(0)
columns=["data_name", "mult_name", "k", "num_queries", "avg_per_same", "latency"]
empty_results = lambda : pd.DataFrame(columns=columns).astype({"data_name": str, "mult_name": str, "k": int, "num_queries": int, "avg_per_same": float})

out_scale= 1 
out_offset_sum=0
def _setup_task_est(embeddings, queries, ncodebooks, lutconsts):
  hparams_dict = {'ncodebooks': ncodebooks, 'lut_work_const': lutconsts}
  est = vq_amm.MithralMatmul(**hparams_dict)
  t = time.perf_counter()
  est.fit(embeddings, queries.T)
  py_fit_time=time.perf_counter() - t
  est.predict(embeddings, queries[:100].T) # sets luts, scales, offsets. No impact on accuracy since relearned 
  task=mithral_wrapped.mithral_amm_task_float(*embeddings.shape,queries.shape[1], ncodebooks, lutconsts)
  task.amm.out_mat = np.zeros(task.amm.out_mat.shape)
  task.X=embeddings
  copy_python_to_amm(est, task.amm)
  copy_python_luts(est, task.amm)
  print('made scale', task.amm.out_scale, task.amm.out_offset_sum)
  global out_scale, out_offset_sum
  out_scale = task.amm.out_scale
  out_offset_sum=task.amm.out_offset_sum
  return task,est 
  
def mithral_mult(task, E,Q):
  D,M = Q.shape
  task.Q=Q
  task.amm.M = M
  t = time.perf_counter()
  task.amm.out_scale = out_scale
  task.amm.out_offset_sum = out_offset_sum 
  task.run_matmul(True) # if true this changes out_scale and out_offset_sum; possibly to invalid/bad reasons TODO
  Y_hat=task.amm.out_mat[:,:M] #raw out_mat if just care about relative order for predicting output. slice for test shape used
  Y_hat=(Y_hat.astype(np.uint16)*task.amm.ncodebooks/task.amm.out_scale) + task.amm.out_offset_sum
  #Y_hat=(Y_hat.astype(np.uint16)*task.amm.ncodebooks/out_scale) + out_offset_sum
  latency=time.perf_counter() - t
  #Y_hat=(Y_hat.astype(np.float64)*task.amm.ncodebooks/task.amm.out_scale) + task.amm.out_offset_sum
  #print(f"1-r2 {1-r2_score(o, Y_hat)}")
  return Y_hat,latency#[:len()]

def est_mult(_est, E,Q):
  hparams_dict = {'ncodebooks': _est.ncodebooks, 'lut_work_const': -1}
  est = vq_amm.MithralMatmul(**hparams_dict)
  #D,M = Q.shape
  #est.enc.M = M
  est.fit(E,Q)
  t = time.perf_counter()
  Y_hat=est.predict(E,Q)
  latency=time.perf_counter() - t
  return Y_hat,latency

def _np_dot(  E,Q):
  t = time.perf_counter() 
  mult = np.dot(E,Q)
  latency=time.perf_counter() - t
  return mult,latency
  
def compare_on_emb_queries_by_ix(embeddings, queries,data_name, NREPS,num_queries,ks, ncodebooks=8,lutconsts=-1):
  """Correct if get the same index of the embedding as the query"""
  pass

seed=75 #=52 give 55% right, about 1/4 give <60% right 
def compare_on_emb_queries_by_class(embeddings, queries,data_name, NREPS,num_queries,ks, ncodebooks=8,lutconsts=-1):
  """ Where the class is the largest ix of row of embedding"""
  global seed
  results = empty_results() 

  embeddings_lengths = np.linalg.norm(embeddings, axis=1)
  #embeddings_class=np.apply_along_axis(np.argmax, 1, queries)

  def calc_avg_per_same(k, closest_embeddings_ixs, query_classes):
    """Class is column ix which has largest value in row.
    For each query, get the k closest embeddings and see how many are in the same class as the query.
    """
    assert(num_queries == len(query_classes))
    def _per_same_class(embedding_ixs, query_class):
      classes = np.apply_along_axis(np.argmax, 1, embeddings[embedding_ixs])
      return 100*np.mean(classes == query_class)
    
    return sum(map(lambda q_ix: _per_same_class(closest_embeddings_ixs[-k:, q_ix], query_classes[q_ix]), 
                   range(num_queries))
               )/num_queries

  task,est = _setup_task_est(embeddings, queries, ncodebooks, lutconsts)
  
  #for mult_method, mult_name in ((partial(mithral_mult,task), 'mithral'),):
  for mult_method, mult_name in ((_np_dot, 'numpy'), (partial(est_mult, est), 'py_est'), (partial(mithral_mult,task), 'cpp_mithral')):
  #for mult_method, mult_name in ((_np_dot, 'numpy'), (mithral_mult, 'mithral')):
    np.random.seed(seed)
    seed += 1
    avg_per_by_k=defaultdict(list)
    for _ in range(NREPS):
      rand_ix=np.random.choice(queries.shape[0], num_queries, replace=True)
      search = queries[rand_ix, :]
      search_classes = np.apply_along_axis(np.argmax, 1, search).reshape(-1)
      
      dot_es, latency=mult_method(embeddings,search.T)
      closest_embeddings_ixs = np.argsort( # argsort is run separately for each query
                            np.apply_along_axis(lambda col: col/embeddings_lengths, 
                                                0, 
                                                dot_es),
                            axis=0)
      for k in ks:
        avg_per_same = calc_avg_per_same(k, closest_embeddings_ixs, search_classes)
        if k == ks[-1]:
          if avg_per_same <= 60:
              print("BAD", seed)
          print(f"Avg Percent of top-{k} in query's class {avg_per_same:.2f}% for {data_name} by method {mult_name} on {num_queries} queries")
        avg_per_by_k[k].append(avg_per_same)
    rep_of_avgs = {k:f"{sum(avgs)/len(avgs):.2f} SD {np.std(avgs):.2f}" for k, avgs in avg_per_by_k.items()}
    print(f"For {data_name} computed by {mult_name} Over {NREPS} of {num_queries} queries by k")
    pprint(rep_of_avgs)
    
    new = pd.DataFrame([[data_name, mult_name, k, num_queries, avg_per_same, latency] 
                        for k, avgs in avg_per_by_k.items()
                        for avg_per_same in avgs], 
                       columns=columns)
    results=pd.concat([results,new], ignore_index=True)
        
  results['data_name'] = results.data_name.astype('category')
  results['mult_name'] = results.mult_name.astype('category')
  return results

def summary_plot(results, ncodebooks, name="", save=False):
  now=datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
  _dir = os.path.dirname(os.path.abspath(__file__))
  acc_path=os.path.join(_dir, '..', 'experiments', 'results', 'embeddings', f'accuaracy_{name}_{now}.png')
  lat_path=os.path.join(_dir, '..', 'experiments', 'results', 'embeddings', f'latency_{name}_{now}.png')
  
  g_acc = sns.catplot(data = results, y='avg_per_same', x="mult_name", hue= "data_name",col='k', aspect=0.5, kind='swarm')
  g_acc.fig.subplots_adjust(top=0.9) 
  g_acc.fig.suptitle(f"Percent of top-k closest embeddings in same class as query {name}")
  
  # no need to seperate by k, all latencies the same
  g_lat = sns.catplot(data = results.query("k==1"), y='latency', x="mult_name", hue= "data_name", aspect=0.5, kind='swarm')
  g_lat.fig.subplots_adjust(top=0.9) 
  g_lat.fig.suptitle(f"Latency of computing {num_queries} queries with {ncodebooks} {name}")
  if save:
    g_acc.fig.savefig(acc_path) 
    print(acc_path)
    g_lat.fig.savefig(lat_path) 
    print(lat_path)
  
num_queries=32 #512*8
NREPS = 5*100
ncodebooks=16
if True: 
  ks=[1,5,10,100,1000]
  name='clip_text_q_img_ix'
  text_emb_subset = text_emb[np.random.choice(text_emb.shape[0], 10000, replace=True)]
  # use by ix
  results=compare_on_emb_queries_by_class(img_emb, text_emb_subset,name, NREPS, num_queries,ks,ncodebooks=8)
  summary_plot(results, ncodebooks, name=name, save=True)
else:
  ks=[1,5,10,100]
  results = empty_results() 
  for data in itertools.chain(*data_sources[1:]):
    [W_test,W_train, X_test, X_train, Y_test, Y_train] = attrgetter('W_test','W_train', 'X_test', 'X_train', 'Y_test', 'Y_train')(data)
    
    #ncodebooks=2**math.floor(np.log2(Y_train.shape[1])) #>16 errors on cifar100
    queries = Y_test[:-(len(Y_test)%32)]
    embeddings = Y_train[:-(len(Y_train)%32)]
    new=compare_on_emb_queries_by_class(embeddings, queries, data.name, NREPS, num_queries,ks,ncodebooks=ncodebooks)
    results = pd.concat([new, results],ignore_index=True)
  summary_plot(results, ncodebooks, name=data.name, save=True)
#%% ###     Scrap
#results.to_csv("py_v_cpp_mithral_for_acc_on_cifar100.csv")

gb = results.groupby(['mult_name', 'k'])['avg_per_same']
mn=gb.mean().unstack('k').loc[['numpy', 'cpp_mithral', 'py_est']]
sd=gb.std().unstack('k').loc[['numpy', 'cpp_mithral', 'py_est']]
se=gb.sem().unstack('k').loc[['numpy', 'cpp_mithral', 'py_est']]
diff_se = (mn.loc['py_est'] - mn.loc['cpp_mithral'])/np.sqrt(se.loc['py_est']**2 + se.loc['cpp_mithral']**2)
print(
  mn,sd,se,diff_se,
  sep='\n'
)

# # py v cpp_mithral is about the same in terms of accuracy, esp as py fits and then predicts
# # For 500 reps of 16 codebooks of 32 queries on cifar100; scales/offset calc'd in mithral; each mult fn ran on exact same data
# #		mean
# k                 1         5          10         100
# mult_name                                            
# numpy        90.96875  90.65125  90.458750  89.576313
# cpp_mithral  83.53750  84.46250  84.751875  84.774375
# py_est       84.55625  85.19250  85.211250  84.805437
# # 	sd
# k                 1         5         10        100
# mult_name                                          
# numpy        5.198628  4.479247  4.338717  4.350878
# cpp_mithral  6.377225  5.259442  4.998312  4.715984
# py_est       6.554634  5.223326  4.944256  4.625267
# #		standard error 
# k                 1         5         10        100
# mult_name                                          
# numpy        0.232490  0.200318  0.194033  0.194577
# cpp_mithral  0.285198  0.235209  0.223531  0.210905
# py_est       0.293132  0.233594  0.221114  0.206848
# # Z score of py_est's standard error being larger than cpp_mithral 
# k
# 1      2.490955
# 5      2.202137
# 10     1.461041
# 100    0.105150

#%%
k=100
closest_embeddings_ix = np.argsort((embeddings @ query.T).reshape(-1) / embeddings_lengths)
ct= defaultdict(int)
for ix,a in enumerate(embeddings[closest_embeddings_ix[-k:]]):
  cosine_sim=np.dot(a, query.reshape(-1))/ (np.linalg.norm(a) * np.linalg.norm(query))
  embedding_class =np.argmax(a)
  print(cosine_sim, embedding_class, search_classes)
  ct[embedding_class] += 1
print(f"Percent of top-{k} in same class as query {100*ct[search_classes[0]]/k:.2f}%")

#%%
t = time.perf_counter()

lutconsts=-1
np_time=time.perf_counter() - t