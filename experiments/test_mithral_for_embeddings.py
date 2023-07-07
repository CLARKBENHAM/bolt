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
from sklearn.metrics import r2_score, roc_curve, roc_auc_score, log_loss
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
text_emb = text_emb[:1024]
img_emb = img_emb[:1024]
MetricsSoftmax = namedtuple("MetricsSoftmax", ["np_time", "py_fit_time", "py_est_time", "py_est_r2", "py_est_per_ix_kept", "copy_to_cpp_time", "cpp_est_time", "cpp_est_r2", "cpp_est_per_ix_kept"])

# How do you measure the accuracy of embedding search?
# 1. How many of the top-k closest embeddings are in the same class as the query embedding?


#%% # Utils
acc_columns=["data_name", "mult_name", "k", "num_queries", "avg_per_same", "latency"]
empty_acc_results = lambda : pd.DataFrame(columns=acc_columns).astype({"data_name": str, "mult_name": str, "k": int, "num_queries": int, "avg_per_same": float, "latency": float})
dist_types = {"data_name": str, "mult_name": str, "num_queries": int, "ncodebooks": int, "cosine": float, "l2": float, "l1": float, "latency": float}
dist_columns = list(dist_types.keys())
empty_dist_results = lambda : pd.DataFrame(columns=dist_columns).astype(dist_types)
seed=75
num_queries=32 #512*8
NREPS = 5
ncodebooks=16
ks=[1,5,10,100]

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
  task.encode()
  #print('made scale', task.amm.out_scale, task.amm.out_offset_sum)
  #global out_scale, out_offset_sum
  #out_scale = task.amm.out_scale
  #out_offset_sum=task.amm.out_offset_sum
  return task,est 
  
def mithral_mult(task, E,Q):
  D,M = Q.shape
  task.Q=Q
  task.amm.M = M
  #task.amm.out_scale = out_scale
  #task.amm.out_offset_sum = out_offset_sum 
  scale,sum = task.amm.out_scale, task.amm.out_offset_sum
  t = time.perf_counter()
  #task.run_matmul(True) # if true this changes out_scale and out_offset_sum; possibly to invalid/bad reasons TODO
  #task.lut()
  task.scan()
  Y_hat=task.amm.out_mat[:,:M] #raw out_mat if just care about relative order for predicting output. slice for test shape used
  Y_hat=(Y_hat.astype(np.uint16)*task.amm.ncodebooks/task.amm.out_scale) + task.amm.out_offset_sum
  #Y_hat=(Y_hat.astype(np.uint16)*task.amm.ncodebooks/out_scale) + out_offset_sum
  latency=time.perf_counter() - t
  print(f"before: {(scale,sum)} \n after: {(task.amm.out_scale, task.amm.out_offset_sum)}")
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
  
def summary_plot_acc(acc_results, ncodebooks, name="", acc_title = None, save=False):
  now=datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
  _dir = os.path.dirname(os.path.abspath(__file__))
  acc_path=os.path.join(_dir, '..', 'experiments', 'results', 'embeddings', f'accuaracy_{name}_{now}.png')
  lat_path=os.path.join(_dir, '..', 'experiments', 'results', 'embeddings', f'latency_{name}_{now}.png')
  
  g_acc = sns.catplot(data = acc_results, y='avg_per_same', x="mult_name", hue= "data_name",col='k', aspect=0.5, kind='swarm')
  g_acc.fig.subplots_adjust(top=0.9) 
  g_acc.fig.suptitle( acc_title or f"Percent of top-k closest embeddings in same class as query {name}")
  
  # no need to seperate by k, all latencies the same
  g_lat = sns.catplot(data = acc_results.query("k==1"), y='latency', x="mult_name", hue= "data_name", aspect=0.7, kind='swarm')
  g_lat.fig.subplots_adjust(top=0.9) 
  g_lat.fig.suptitle(f"Latency of computing {num_queries} queries with {ncodebooks} {name}")
  if save:
    g_acc.fig.savefig(acc_path) 
    print(acc_path)
    g_lat.fig.savefig(lat_path) 
    print(lat_path)

def summary_plot_dist(dist_results, save=False):
  ncodebooks = dist_results['ncodebooks'].unique()
  assert(len(ncodebooks)==1)
  ncodebooks = ncodebooks[0]
  num_queries = dist_results['num_queries'].unique()
  assert(len(num_queries)==1)
  num_queries = num_queries[0]
  lat_results = dist_results.iloc[::num_queries, :]# unique latencies
  
  write_data = [] 
  fig = plt.figure()
  lat_ax = sns.swarmplot(data = lat_results , x="mult_name" ,y='latency' ,  hue= "data_name", legend=False)
  sns.boxplot(data = lat_results , x="mult_name", y='latency' , hue= "data_name",
          ax=lat_ax,
          showcaps=False,boxprops={'facecolor':'None'},
          showfliers=False,whiskerprops={'linewidth':0},
          )
  plt.suptitle(f"Latency of computing {num_queries} queries with {ncodebooks} codebooks")
  write_data += [(fig, 'latency')]
  plt.show()
  
  for y in ['cosine', 'l2', 'l1']:
    fig = plt.figure()
    g_ax = sns.swarmplot(data = dist_results , x="mult_name", y=y , hue= "data_name" , size=3, alpha=0.7, legend=False)
    sns.boxplot(data = dist_results , x="mult_name" , y=y, hue= "data_name",
            ax=g_ax,
            showcaps=False,boxprops={'facecolor':'None'},
            showfliers=False,whiskerprops={'linewidth':0.5}, linewidth=0.7,
            )
    plt.suptitle(f"{y} {'similarity' if y=='cosine' else 'distance'} of correct and returned embedding ({ncodebooks} codebooks)")
    plt.gca().legend().set_visible(False)
    write_data += [(fig, y)]
    plt.show()
  
  if save:
    now=datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    _dir = os.path.dirname(os.path.abspath(__file__))
    for g, name in write_data:
      path=os.path.join(_dir, '..', 'experiments', 'results', 'embeddings', f'distance_comp_{name}_{now}.png')
      g.savefig(path) 
      print(path)

#%%  How accurate is getting Class of Cifar-100
def compare_on_emb_queries_by_class(embeddings, queries,data_name, NREPS,num_queries,ks, ncodebooks=8,lutconsts=-1,seed=seed):
  """ 
  Where the class is the largest ix of row of embedding
  How often in top K is the correct vector returned?
  """
  def calc_avg_per_same(k, closest_embeddings_ixs, query_classes):
    """Class is column ix which has largest value in row.
    For each query, get the k closest embeddings and see how many are in the same class as the query.
    """
    def _per_same_class(embedding_ixs, query_class):
      classes = np.apply_along_axis(np.argmax, 1, embeddings[embedding_ixs])
      return 100*np.mean(classes == query_class)
    
    return sum(map(lambda q_ix: _per_same_class(closest_embeddings_ixs[-k:, q_ix], query_classes[q_ix]), 
                   range(num_queries))
               )/num_queries

  results = empty_acc_results()
  embeddings_lengths = np.linalg.norm(embeddings, axis=1)

  task,est = _setup_task_est(embeddings, queries, ncodebooks, lutconsts)
  
  for mult_method, mult_name in ((_np_dot, 'numpy'), (partial(est_mult, est), 'py_est'), (partial(mithral_mult,task), 'cpp_mithral')):
  #for mult_method, mult_name in ((partial(mithral_mult,task), 'cpp_mithral'),):
    np.random.seed(seed)
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
              print("BAD")
          print(f"Avg Percent of top-{k} in query's class {avg_per_same:.2f}% for {data_name} by method {mult_name} on {num_queries} queries")
        avg_per_by_k[k].append(avg_per_same)
    rep_of_avgs = {k:f"{sum(avgs)/len(avgs):.2f} SD {np.std(avgs):.2f}" for k, avgs in avg_per_by_k.items()}
    print(f"For {data_name} computed by {mult_name} Over {NREPS} of {num_queries} queries by k")
    pprint(rep_of_avgs)
    
    new = pd.DataFrame([[data_name, mult_name, k, num_queries, avg_per_same, latency] 
                        for k, avgs in avg_per_by_k.items()
                        for avg_per_same in avgs], 
                       columns=acc_columns)
    results=pd.concat([results,new], ignore_index=True)
        
  results['data_name'] = results.data_name.astype('category')
  results['mult_name'] = results.mult_name.astype('category')
  return results

acc_results = empty_acc_results() 
for data in itertools.chain(*data_sources[1:]):
  [W_test,W_train, X_test, X_train, Y_test, Y_train] = attrgetter('W_test','W_train', 'X_test', 'X_train', 'Y_test', 'Y_train')(data)
  queries = Y_test[:-(len(Y_test)%32)]
  embeddings = Y_train[:-(len(Y_train)%32)]
  new=compare_on_emb_queries_by_class(embeddings, queries, data.name, NREPS, num_queries,ks,ncodebooks=ncodebooks, seed=seed)
  acc_results = pd.concat([new, acc_results],ignore_index=True)

summary_plot_acc(acc_results, ncodebooks, name=data.name, save=True)

#acc_results.to_csv("py_v_cpp_mithral_for_acc_on_cifar100.csv")
#%%
gb = acc_results.groupby(['mult_name', 'k'])['avg_per_same']
mn=gb.mean().unstack('k').loc[['numpy', 'cpp_mithral', 'py_est']]
sd=gb.std().unstack('k').loc[['numpy', 'cpp_mithral', 'py_est']]
se=gb.sem().unstack('k').loc[['numpy', 'cpp_mithral', 'py_est']]
diff_se = (mn.loc['py_est'] - mn.loc['cpp_mithral'])/np.sqrt(se.loc['py_est']**2 + se.loc['cpp_mithral']**2)
l = acc_results.query("k==1").groupby('mult_name')['latency'].mean()
l_sd = acc_results.query("k==1").groupby('mult_name')['latency'].std()

print(
  f"mean:\n{mn}",
  f"standard deviation:\n{sd}",
  f"standard error:\n{se}",
  f"py-cpp mithral Z-score:\n{diff_se}",
  f"latency:\n{l}",
  f"latency SD:\n{l_sd}",
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
# # Z score of py_est's mean being larger than cpp_mithral 
# k
# 1      2.490955
# 5      2.202137
# 10     1.461041
# 100    0.105150

#%%
def compare_on_emb_retrieval(embeddings, queries,data_name, NREPS,num_queries,ks, ncodebooks=8,lutconsts=-1, seed=seed):
  """
  How often the embedding matching a query (as determined by ix) retrieved in top k
  """
  results = empty_acc_results() 

  embeddings_lengths = np.linalg.norm(embeddings, axis=1)

  def calc_avg_right_returned(k, closest_embeddings_ixs, query_classes):
    """
    How often the embedding matching a query (as determined by ix) retrieved in top k
    """
    return 100*np.mean([np.sum(closest_embeddings_ixs[-k:, q_ix]==query_classes[q_ix])
                    for q_ix in range(query_classes.shape[0])])
    
  task,est = _setup_task_est(embeddings, queries, ncodebooks, lutconsts)
  
  for mult_method, mult_name in ((_np_dot, 'numpy'), (partial(est_mult, est), 'py_est'), (partial(mithral_mult,task), 'cpp_mithral')):
    np.random.seed(seed)
    avg_per_by_k=defaultdict(list)
    for _ in range(NREPS):
      rand_ix = np.random.choice(queries.shape[0], num_queries, replace=True)
      search = queries[rand_ix, :]
      search_classes = rand_ix
      search_lengths = np.linalg.norm(search, axis=1)
      dot_es, latency=mult_method(embeddings,search.T)
      normalized_cosine = np.apply_along_axis(lambda row: row/search_lengths, 
                                              1,
                                              np.apply_along_axis(lambda col: col/(embeddings_lengths), 
                                                0, 
                                                dot_es))
      closest_embeddings_ixs = np.argsort(normalized_cosine, axis=0)
      
      for k in ks:
        avg_per_same = calc_avg_right_returned(k, closest_embeddings_ixs, search_classes)
        if k == ks[0]:
          print(f"Avg Percent top-{k} found query's matching embedding {avg_per_same:.2f}% for {data_name} by method {mult_name} on {num_queries} queries")
        avg_per_by_k[k].append(avg_per_same)
    rep_of_avgs = {k:f"{sum(avgs)/len(avgs):.2f} SD {np.std(avgs):.2f}" for k, avgs in avg_per_by_k.items()}
    print(f"For {data_name} computed by {mult_name} Over {NREPS} reps of {num_queries} queries by k")
    pprint(rep_of_avgs)
    
    new = pd.DataFrame([[data_name, mult_name, k, num_queries, avg_per_same, latency] 
                        for k, avgs in avg_per_by_k.items()
                        for avg_per_same in avgs], 
                       columns=acc_columns)
    results=pd.concat([results,new], ignore_index=True)
        
  results['data_name'] = results.data_name.astype('category')
  results['mult_name'] = results.mult_name.astype('category')
  return results

for e,q, name in [(img_emb, img_emb, "img_queries_img_ix"),
                  (img_emb, text_emb, "text_queries_img_ix"),
                  (text_emb, text_emb, "text_queries_text_ix")]:
  acc_results=compare_on_emb_retrieval(q,e,name, NREPS, num_queries,ks,ncodebooks=ncodebooks)
  summary_plot_acc(acc_results, ncodebooks, name=name, acc_title = f"Avg% the Closest Embedding is in Top-K {name}", save=True)

#%%
def compare_dist_ret_from_true(embeddings, queries,data_name, NREPS, num_queries, ncodebooks=8,lutconsts=-1, seed=seed):

  """
    Using different methods, how far are returned embeddings from the true embedding
    The true embedding is the one that shares an index with the query the query
  """
  embeddings_lengths = np.linalg.norm(embeddings, axis=1)
  task,est = _setup_task_est(embeddings, queries, ncodebooks, lutconsts)
  results = empty_dist_results()
  
  for mult_method, mult_name in ((_np_dot, 'numpy'), (partial(est_mult, est), 'py_est'), (partial(mithral_mult,task), 'cpp_mithral')):
  #for mult_method, mult_name in ((partial(est_mult, est), 'py_est'), (partial(mithral_mult,task), 'cpp_mithral')):
    np.random.seed(seed)
    cos_sim= []
    l2_dis = []
    l1_dis = []
    latencies = []
    for _ in range(NREPS):
      rand_ix = np.random.choice(queries.shape[0], num_queries, replace=True)
      search = queries[rand_ix, :]
      search_classes = rand_ix
      dot_es, latency=mult_method(embeddings,search.T)
      closest_embeddings_ixs = np.argsort( # argsort is run separately for each query
                            np.apply_along_axis(lambda col: col/embeddings_lengths, 
                                                0, 
                                                dot_es),
                            axis=0)
      ret_embs =  embeddings[closest_embeddings_ixs[-1, :]]
      true_embs = embeddings[search_classes]
      
      pairs = [(ret_embs[i], true_embs[i]) for i in range(ret_embs.shape[0])]
      cal_cos_s = lambda pair: np.dot(pair[0], pair[1])/(np.linalg.norm(pair[0])*np.linalg.norm(pair[1]))
      cal_l2_d = lambda pair: np.linalg.norm(pair[0]-pair[1]) 
      cal_l1_d = lambda pair: np.linalg.norm(pair[0]-pair[1], ord=1) 
      cos_sim+= list(map(cal_cos_s, pairs))
      l2_dis += list(map(cal_l2_d, pairs)) 
      l1_dis += list(map(cal_l1_d, pairs)) 
      latencies += [latency]*len(pairs)

    new = pd.DataFrame([[data_name, mult_name, num_queries, ncodebooks, cos, l2,l1, latency] 
                        for cos, l2,l1,latency in zip(cos_sim, l2_dis, l1_dis, latencies)],
                       columns=dist_columns)
    print(f"Method {mult_name} compared returned vs. query of {data_name} dataset on {num_queries} queries") 
    print(new[['cosine', 'l2', 'l1', 'latency']].mean().rename('mean'))
    print(new[['cosine', 'l2', 'l1', 'latency']].std().rename('std'))
    results=pd.concat([results,new], ignore_index=True)
  return results

dist_results = empty_dist_results() 
for e,q, name in [(img_emb, img_emb, "img_queries_img"),
                  (img_emb, text_emb, "text_queries_img"),
                  (text_emb, text_emb, "text_queries_text")]:
  new = compare_dist_ret_from_true(e, q , name, NREPS*20, num_queries , ncodebooks=ncodebooks)
  dist_results = pd.concat([new, dist_results],ignore_index=True)

summary_plot_dist(dist_results, save=True)

#%%  ###     Scrap

def logloss_of_softmax(normalized_cosine, y_true):
  # This isn't a mathemtatically valid way to get probabilities 
  softmax = np.exp(normalized_cosine)/np.sum(np.exp(normalized_cosine), axis=0, keepdims=True)
  assigned_true_prob = softmax[y_true] 
  binary_softmax = np.stack([1-assigned_true_prob, assigned_true_prob],axis=1)
  onezeros = np.stack([np.zeros((num_queries)), np.ones((num_queries))],axis=1)
  logloss = log_loss(onezeros, binary_softmax)
  return logloss

def plot_dists_togther(results):
  result_columns = ['cosine', 'l2', 'l1', 'latency']
  metadata_columns = [c for c in dist_columns if c not in  result_columns]
  data = pd.DataFrame(columns = metadata_columns + ['measure', 'value'])
  for res_col in result_columns:
    sub = results[metadata_columns + [res_col]].copy()
    sub['measure'] = res_col 
    sub['value'] = sub[res_col]
    del sub[res_col]
    data = pd.concat([data, sub], ignore_index=True)
  
  g = sns.FacetGrid(data, col="measure", height=4, aspect=.5)
  g.map(sns.barplot, "mult_name", "value")
  
  #  g_cos = sns.catplot(data = results , y='cosine'  , x="mult_name" , hue= "data_name" , kind='swarm', size=3, alpha=0.3, legend=False)
  #  g_l2  =  sns.catplot(data = results , y='l2'      , x="mult_name" , hue= "data_name" , kind='swarm', size=3, alpha=0.3, legend=False)
  #  g_l1  = sns.catplot(data = results , y='l1'      , x="mult_name" , hue= "data_name" , kind='swarm', size=3, alpha=0.3, legend=False)
  

#%%

rand_ix = np.random.choice(text_emb.shape[0], 10000, replace=True)
text_emb_subset = text_emb[rand_ix]
text_emb_subset = text_emb[rand_ix]

y_true = np.array([search_classes==i for i in range(queries.shape[0])])
      
#tpr, fpr, t = roc_curve(np.ones((num_queries)), assigned_true_prob)
#auc = roc_auc_score(np.ones((num_queries)), assigned_true_prob)
#plt.plot(fpr,tpr,label=f"{mult_name}, auc={auc}")
      

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