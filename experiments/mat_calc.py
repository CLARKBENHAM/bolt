#!/usr/bin/env python

import numpy as np

from python import vquantizers as vq

if __name__ == '__main__':

        N = 64
        D = 128
        M = 32
        codebooks = 16

        X = np.random.randint(100, size=(N, D)) 
        Q = np.random.randint(100, size=(D, M)) 

        task = vq.MithralEncoder(codebooks)

        task.fit(X)

        X_enc = task.encode_X(X)

        luts, offset, scale = task.encode_Q(Q.T)

        W = task.dists_enc(X_enc, luts, False, offset, scale)

        print("W: ", W)
        print("-----")

        W_real = np.matmul(X, Q)
        mse = np.square(W - W_real).mean()
        ms = np.square(W_real).mean()
        print("mse: ", mse, "; % variance off: ", 100*mse/ms)
        print("offset: ", np.abs(W - W_real))
        print("\n\n")
        print(W, W_real, sep="\n")


