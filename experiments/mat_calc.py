#!/usr/bin/env python

from python import vquantizers as vq
import functools
import numpy as np
import pprint
import scipy
import time
from timeit import default_timer as timer

import cpp

from python import amm
from python import vq_amm
from python import matmul_datasets as md
from python import pyience as pyn
from python import compress

from python import amm_methods as methods
from python import amm_main

from joblib import Memory
_memory = Memory('.', verbose=0)

def mult1(X, Q, codebooks):
    task = vq.MithralEncoder(codebooks)

    task.fit(X)

    X_enc = task.encode_X(X)

    luts, offset, scale = task.encode_Q(Q.T)

    W = task.dists_enc(X_enc, luts, False, offset, scale)

    return W


def compare(X, Q, W):
    #print("W: ", W)
    print("-----")

    W_real = np.matmul(X, Q)
    mse = np.square(W - W_real).mean()
    ms = np.square(W_real).mean()  - np.mean(W_real)**2
    print("mse: ", mse, "; 1-R^2: ", mse/ms)
    print("offset: ", np.abs(W - W_real))
    #print("\n\n")
    #print(W, W_real, sep="\n")



# Flattening out amm_main.py

def _estimator_for_method_id(method_id, **method_hparams):
    """ values to use
    'MithralPQ': vq_amm.MithralPQ,
    'Mithral':   vq_amm.MithralMatmul
    """
    return methods.METHOD_TO_ESTIMATOR[method_id](**method_hparams)


def _fitted_est_for_hparams(method_id, hparams_dict, X_train, W_train,
                            Y_train, **kwargs):
    est = _estimator_for_method_id(method_id, **hparams_dict)
    est.fit(X_train, W_train, Y=Y_train, **kwargs)
    return est


def mult2():
    """combined tests"""
    methods = ['Mithral', 'MithralPQ']
    method_id = methods[0]
    independent_vars = {'niters', 'ncodebooks', 'alpha', 'ncentroids',
                        'd', 'task_id', 'canCheat', 'trial', 'lut_work_const', 'method'}
    # Mithral and MithralPQ hprams
    hparams_dict = [{'ncodebooks': 2, 'lut_work_const': 2}, {'ncodebooks': 2, 'lut_work_const': 4}, {'ncodebooks': 2, 'lut_work_const': -1}, {'ncodebooks': 4, 'lut_work_const': 2}, {'ncodebooks': 4, 'lut_work_const': 4}, {'ncodebooks': 4, 'lut_work_const': -1}, {'ncodebooks': 8, 'lut_work_const': 2}, {'ncodebooks': 8, 'lut_work_const': 4}, {'ncodebooks': 8, 'lut_work_const': -1},
                    {'ncodebooks': 16, 'lut_work_const': 2}, {'ncodebooks': 16, 'lut_work_const': 4}, {'ncodebooks': 16, 'lut_work_const': -1}, {'ncodebooks': 32, 'lut_work_const': 2}, {'ncodebooks': 32, 'lut_work_const': 4}, {'ncodebooks': 32, 'lut_work_const': -1}, {'ncodebooks': 64, 'lut_work_const': 2}, {'ncodebooks': 64, 'lut_work_const': 4}, {'ncodebooks': 64, 'lut_work_const': -1}][0]
    # MithralPQ
    task = functools.partial(md.load_caltech_tasks,
                             filt='dog5x5', limit_ntrain=-1)

    est = _fitted_est_for_hparams(
        method_id, hparams_dict,
        task.X_train, task.W_train, task.Y_train)


def manual_mult_mithral():
    """20x speedup and 0.2% error on random vecs; 50x speedup for n=12800,M=1280"""
    hparams_dict = {'ncodebooks': 8, 'lut_work_const': -1}
    # Timeit in Python
    N = 1280
    D = 128
    M = 1280
    X = np.random.randint(100, size=(N, D))
    W = np.random.randint(100, size=(D, M))
    # X_train, X_test= X[:,:D*3//4], X[:,D*3//4:]
    W_train, W_test = W[:, :M*3//4], W[:, M*3//4:]
    est3 = vq_amm.MithralMatmul(**hparams_dict)
    s = timer()
    est3.fit(X, W_train)
    m = timer()
    Y = X@W_test
    e = timer()
    Y_hat = est3.predict(X, W_test)
    print(np.mean(np.abs(Y_hat-Y)**2)/(np.mean(Y**2)-np.mean(Y)**2),
          f"% Faster: {(e-m)/(m-s)*100}")


def mult_mithral():
    method_id = 'Mithral'
    # guess
    hparams_dict = {'ncodebooks': 8, 'lut_work_const': -1} #fast for tests. 11% test error
    #hparams_dict = {'ncodebooks': 16, 'lut_work_const': -1} #1% test error MSE, 7x faster
    #hparams_dict = {'ncodebooks': 64, 'lut_work_const': -1} 

    task_func = functools.partial(md.load_caltech_tasks,
                             filt='dog5x5', limit_ntrain=-1)
    task = next(task_func())

    est = vq_amm.MithralMatmul(**hparams_dict)
    # y=X@W. #  np.mean(np.abs(Y_hat-task.X_test@task.W_test)**2)/np.mean(np.abs(Y_hat)**2)
    # I assume X is the constant matrix _A in class and B gets encoded each time?
    est.fit(task.X_train, task.W_train, task.Y_train)  # This fit doesn't use Y
    Y_hat = est.predict(task.X_test, task.W_test)
    print(amm_main._compute_metrics(task, Y_hat))

    # Extract Splits and other Hparams
    est2 = vq_amm.MithralMatmul(**hparams_dict)
    fitted = ['ncodebooks', 'ncentroids', 'A_enc', 'luts', 'offset', 'scale']
    est2.__dict__.update(filter(lambda kv: kv[0] in fitted,
                                est.__dict__.items()))
    Y_hat2 = est2.predict(task.X_test, task.W_test)
    print(amm_main._compute_metrics(task, Y_hat2))
    print(np.mean(np.abs(Y_hat-Y_hat2)**2))
    # step into
    Y_hat2 = est2.predict(task.X_test, task.W_test)
    compare(task.X_test, task.W_test, Y_hat2)
     
    X_enc = est.enc.encode_X(task.X_test)
    s = timer()
    # fn to base on? :
    Q_luts, offset, scale = est.enc.encode_Q(task.W_test.T)
    out= est.enc.dists_enc(X_enc, Q_luts, unquantize=True,
                  offset=offset, scale=scale)
    m = timer()
    Y = task.X_test@task.W_test
    e = timer()
    compare(task.X_test, task.W_test, out)
    print(np.mean(np.abs(out-Y)**2)/(np.mean(Y**2)-np.mean(Y)**2),
          f"% Faster: {(e-m)/(m-s)*100}")
    # Fit C++ to these splits 
    # "you'd have to generate the split thresholds, split values, and prototypes in python and pass them to C++."  
    # mithral_encode Looking for: splitdims and all_splitvals
    est.enc.splits_lists #the split lists: are 8 by 4 since there's 4 bits per code?
    #   have within attributes 'dim','offset','scaleby','vals' and fn 'preprocess_x()'
    est.enc.centroids  # aka prototypes
    offsets=est.enc.offsets #? 1 per codebook
    # void mithral_encode(
    # const float* X, int64_t nrows, int ncols,
    # const uint32_t* splitdims, const int8_t* all_splitvals,
    # const float* scales, const float* offsets, int ncodebooks, uint8_t* out)
    X=task.W_test
    nrows=len(X)
    ncols=len(X[0])
    splitdims =[[i.dim for i in l] for l in est.enc.splits_lists]
    all_splitvals= [[i.vals for i in l] for l in est.enc.splits_lists]
    scales=[[i.scaleby for i in l] for l in est.enc.splits_lists] 
    # scales=est.scale
    offsets=[[i.offset for i in l] for l in est.enc.splits_lists] 
    # offsets=est.offset
    ncodebooks=est.ncodebooks
    out=[]
    
     
    # C++ speed + accuracy comparision
     
    # compare(X, Q, out)
    
if __name__ == '__main__':

    N = 128
    D = 128
    M = 128
    codebooks = 32

    X = np.random.randint(100, size=(N, D))
    Q = np.random.randint(100, size=(D, M))

    # W1 = mult1(X, Q, codebooks)
    # compare(X, Q, W1)
    mult_mithral()
    
    
