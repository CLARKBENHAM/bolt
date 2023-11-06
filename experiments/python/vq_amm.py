#!/usr/bin/env python

import abc
import numpy as np

from . import vquantizers as vq
from . import amm

from copy_to_amm import copy_python_to_amm, extract_py_vars, copy_python_luts
import os
import sys
repo_path = os.path.abspath('')[:os.path.abspath('').index('/bolt/') + 6]
sys.path.append(repo_path)
from cpp import mithral_wrapped

KEY_NLOOKUPS = 'nlookups'


class VQMatmul(amm.ApproxMatmul, abc.ABC):
    def __init__(self, ncodebooks, ncentroids=None):
        self.ncodebooks = ncodebooks
        self.ncentroids = (self._get_ncentroids() if ncentroids is None
                           else ncentroids)
        self.enc = self._create_encoder(ncodebooks)
        self.reset_for_new_task()

    @abc.abstractmethod
    def _create_encoder(self, ncodebooks):  # to be overriden by subclasses
        return vq.PQEncoder(ncodebooks=ncodebooks, ncentroids=self.ncentroids,
                            **self._get_encoder_kwargs())

    # @abc.abstractmethod
    def _get_ncentroids(self):
        pass

    @abc.abstractmethod
    def get_speed_metrics(self, A, B, fixedA=False, fixedB=False):
        pass

    def _get_encoder_kwargs(self):  # to be overriden by subclasses
        return {}

    def reset_for_new_task(self):
        self.A_enc = None
        self.luts = None

    def fit(self, A, B, Y=None):
        _, D = A.shape
        if D < self.ncodebooks:
            raise amm.InvalidParametersException(
                'D < C: {} < {}'.format(D, self.ncodebooks))
        self.enc.fit(A, B.T)

    def set_A(self, A):
        self.A_enc = self.enc.encode_X(A)

    def set_B(self, B):
        self.luts = self.enc.encode_Q(B.T)

    def __call__(self, A, B):
        if self.A_enc is None:
            self.set_A(A)
        if self.luts is None:
            self.set_B(B)
        return self.enc.dists_enc(self.A_enc, self.luts)

    def get_params(self):
        return {'ncodebooks': self.ncodebooks}


# ================================================================ PQ

class PQMatmul(VQMatmul):

    def _create_encoder(self, ncodebooks):  # to be overriden by subclasses
        return vq.PQEncoder(ncodebooks=ncodebooks, ncentroids=self.ncentroids,
                            **self._get_encoder_kwargs())

    def _get_ncentroids(self):
        return 256

    def get_speed_metrics(self, A, B, fixedA=False, fixedB=False):
        # data encoding and LUT costs
        nmuls = 0
        nmuls += 0 if fixedA else A.shape[0] * A.shape[1] * self.ncentroids
        nmuls += 0 if fixedB else B.shape[0] * B.shape[1] * self.ncentroids
        nlookups = A.shape[0] * B.shape[1] * self.ncodebooks
        return {amm.KEY_NMULTIPLIES: nmuls, KEY_NLOOKUPS: nlookups}


# ================================================================ OPQ

class OPQMatmul(PQMatmul):

    def _get_encoder_kwargs(self):
        return dict(preproc='OPQ')

    def get_speed_metrics(self, A, B, fixedA=False, fixedB=False):
        metrics = super().get_speed_metrics(A, B, fixedA=fixedA, fixedB=fixedB)
        rot_nmuls = A.shape[0] * A.shape[1] * A.shape[1]  # OPQ rotation cost
        metrics[amm.KEY_NMULTIPLIES] += rot_nmuls
        return metrics


# ================================================================ Bolt

class BoltMatmul(PQMatmul):

    # def __init__(self, ncodebooks):
    #     self.ncodebooks = ncodebooks
    #     self.ncentroids = 16
    #     self.enc = self._create_encoder(self.ncodebooks)
    #     self._reset()

    def _get_ncentroids(self):
        return 16

    def _create_encoder(self, ncodebooks):
        return vq.PQEncoder(ncodebooks=ncodebooks, ncentroids=self.ncentroids,
                            quantize_lut=True,
                            # quantize_lut=False,
                            # accumulate_how='mean',
                            accumulate_how='sum',
                            upcast_every=-1,
                            # upcast_every=2,
                            # upcast_every=4,
                            # upcast_every=256,  # fine as long as using mean
                            # TODO set quantize_lut=True after debug
                            **self._get_encoder_kwargs())


class GEHTBoltMatmul_CovTopk(BoltMatmul):

    def _get_encoder_kwargs(self):
        return dict(
            preproc='GEHT', sample_how='deterministic', stats_mat='cov')


class GEHTBoltMatmul_CovSamp(BoltMatmul):

    def _get_encoder_kwargs(self):
        return dict(
            preproc='GEHT', sample_how='importance', stats_mat='cov')


class GEHTBoltMatmul_CorrTopk(BoltMatmul):

    def _get_encoder_kwargs(self):
        return dict(
            preproc='GEHT', sample_how='deterministic', stats_mat='corr')


class GEHTBoltMatmul_CorrSamp(BoltMatmul):

    def _get_encoder_kwargs(self):
        return dict(
            preproc='GEHT', sample_how='importance', stats_mat='corr')


class BoltSplits(BoltMatmul):

    def _get_encoder_kwargs(self):
        return dict(
            preproc='PQ', encode_algo='splits')

    def get_speed_metrics(self, A, B, fixedA=False, fixedB=False):
        metrics = super().get_speed_metrics(A, B, fixedA=fixedA, fixedB=fixedB)
        nmuls = 0
        nmuls += 0 if fixedB else B.shape[0] * B.shape[1] * self.ncentroids
        metrics[amm.KEY_NMULTIPLIES] = nmuls
        return metrics


class BoltMultiSplits(BoltMatmul):

    def _get_encoder_kwargs(self):
        return dict(encode_algo='multisplits')

    def get_speed_metrics(self, A, B, fixedA=False, fixedB=False):
        metrics = super().get_speed_metrics(A, B, fixedA=fixedA, fixedB=fixedB)
        nmuls = 0
        nmuls += 0 if fixedB else B.shape[0] * B.shape[1] * self.ncentroids
        metrics[amm.KEY_NMULTIPLIES] = nmuls
        return metrics


class BoltPermMultiSplits(BoltMatmul):

    def _get_encoder_kwargs(self):
        return dict(preproc='GEHT', encode_algo='multisplits')

    def get_speed_metrics(self, A, B, fixedA=False, fixedB=False):
        metrics = super().get_speed_metrics(A, B, fixedA=fixedA, fixedB=fixedB)
        nmuls = 0
        nmuls += 0 if fixedB else B.shape[0] * B.shape[1] * self.ncentroids
        metrics[amm.KEY_NMULTIPLIES] = nmuls
        return metrics


class PQPerm(PQMatmul):

    def _get_encoder_kwargs(self):
        return dict(preproc='GEHT')

    def get_speed_metrics(self, A, B, fixedA=False, fixedB=False):
        metrics = super().get_speed_metrics(A, B, fixedA=fixedA, fixedB=fixedB)
        nmuls = 0
        nmuls += 0 if fixedB else B.shape[0] * B.shape[1] * self.ncentroids
        metrics[amm.KEY_NMULTIPLIES] = nmuls
        return metrics


class PQMultiSplits(PQMatmul):

    def __init__(self, ncodebooks, ncentroids=256):
        super().__init__(ncodebooks=ncodebooks, ncentroids=ncentroids)

    def _get_encoder_kwargs(self):
        return dict(encode_algo='multisplits')

    def get_params(self):
        return {'ncodebooks': self.ncodebooks, 'ncentroids': self.ncentroids}

    def get_speed_metrics(self, A, B, fixedA=False, fixedB=False):
        metrics = super().get_speed_metrics(A, B, fixedA=fixedA, fixedB=fixedB)
        nmuls = 0
        nmuls += 0 if fixedB else B.shape[0] * B.shape[1] * self.ncentroids
        metrics[amm.KEY_NMULTIPLIES] = nmuls
        return metrics


class PQPermMultiSplits(PQMatmul):

    def __init__(self, ncodebooks, ncentroids=256):
        super().__init__(ncodebooks=ncodebooks, ncentroids=ncentroids)

    def _get_encoder_kwargs(self):
        return dict(preproc='GEHT', encode_algo='multisplits')

    def get_params(self):
        return {'ncodebooks': self.ncodebooks, 'ncentroids': self.ncentroids}

    def get_speed_metrics(self, A, B, fixedA=False, fixedB=False):
        metrics = super().get_speed_metrics(A, B, fixedA=fixedA, fixedB=fixedB)
        nmuls = 0
        nmuls += 0 if fixedB else B.shape[0] * B.shape[1] * self.ncentroids
        metrics[amm.KEY_NMULTIPLIES] = nmuls
        return metrics


# ================================================================ Mithral

class OldMithralPQ(PQMatmul):

    # def _get_ncentroids(self):
    #     return 16

    def __init__(self, ncodebooks):
        super().__init__(ncodebooks=ncodebooks, ncentroids=16)

    def _create_encoder(self, ncodebooks):
        return vq.PQEncoder(ncodebooks=ncodebooks, ncentroids=self.ncentroids,
                            encode_algo='multisplits',
                            quantize_lut=True,
                            upcast_every=16,  # fine as long as using mean
                            accumulate_how='mean')

    def get_speed_metrics(self, A, B, fixedA=False, fixedB=False):
        N, D = A.shape
        D, M = B.shape
        # data encoding and LUT costs
        nmuls = 0
        nmuls += 0 if fixedA else N * D  # offset + scale before quantize
        nmuls += 0 if fixedB else M * self.ncentroids * D
        # lookups given encoded data + luts
        nlookups = N * M * self.ncodebooks
        return {amm.KEY_NMULTIPLIES: nmuls, KEY_NLOOKUPS: nlookups}


class MithralMatmul(VQMatmul):

    def __init__(self, ncodebooks, lut_work_const=-1):
        self.lut_work_const = lut_work_const
        if (lut_work_const is not None) and (lut_work_const > 0) and (
                lut_work_const > ncodebooks):
            raise amm.InvalidParametersException(
                "lut_work_const > ncodebooks: {} > {}".format(
                    lut_work_const, ncodebooks))
        super().__init__(ncodebooks=ncodebooks, ncentroids=16)

    # def _get_ncentroids(self):
    #     return 16

    # def fit(self, A, B, Y=None):
    #     super().fit(self, A, B, Y=Y)

    def _create_encoder(self, ncodebooks):
        return vq.MithralEncoder(
            ncodebooks=ncodebooks, lut_work_const=self.lut_work_const)

    def get_params(self):
        return {'ncodebooks': self.ncodebooks,
                'lut_work_const': self.lut_work_const}

    def get_speed_metrics(self, A, B, fixedA=False, fixedB=False):
        N, D = A.shape
        D, M = B.shape
        # data encoding and LUT costs
        nmuls = 0
        nmuls += 0 if fixedA else N * D  # offset + scale before quantize
        nmuls_per_codebook_per_output = self.ncentroids * D
        nmuls_per_output = nmuls_per_codebook_per_output * self.ncodebooks
        nmuls += 0 if fixedB else nmuls_per_output * M
        # lookups given encoded data + luts
        nlookups = N * M * self.ncodebooks
        return {amm.KEY_NMULTIPLIES: nmuls, KEY_NLOOKUPS: nlookups}

    def set_B(self, B):
        self.luts, self.offset, self.scale = self.enc.encode_Q(B.T)

    #What outputs Y_hat matrix 
    def __call__(self, A, B):
        if self.A_enc is None:
            self.set_A(A)
        if self.luts is None:
            self.set_B(B)
        return self.enc.dists_enc(self.A_enc, self.luts,
                                  offset=self.offset, scale=self.scale)


class MithralPQ(MithralMatmul):

    def __init__(self, ncodebooks):
        super().__init__(ncodebooks=ncodebooks, lut_work_const=1)

### Calling Wrapped C++
class MithralMatmulCppImp(MithralMatmul):
    # How do I set A and B seperately from each other? I'd have to change C++ code
    def __init__(self, ncodebooks, keep_cpp_type=False, task_type='mithral_amm_float', create_lut_at_test=False, lut_work_const=-1):
        """
        Args:
            keep_cpp_type: if the C++ output type should be kept. Else the output is converted to a float (slower)
                Set true if only care about the relative order amoung columns for predicting category
            task_type (str, optional): _description_. Defaults to 'mithral_amm_float', where '_float' is the input type expected
            create_lut_at_test (bool, optional): If to re-create luts using test time values of B
                Defaults to False.
        """
        super().__init__(ncodebooks=ncodebooks, lut_work_const=lut_work_const)
        self.keep_cpp_type=keep_cpp_type
        self.task_type =task_type
        self.create_lut_at_test=create_lut_at_test
        if task_type=='mithral_amm_float':
            self.make_task = mithral_wrapped.mithral_amm_task_float
            self.task=None
        else:
            assert(f'bad task_type {task_type}') 
    
    def fit(self, A, B, Y=None):
        N, D = A.shape
        _,M = B.shape
        #if D < self.ncodebooks:
        #    raise amm.InvalidParametersException(
        #        'D < C: {} < {}'.format(D, self.ncodebooks))
        #self.enc.fit(A, B.T)
        super().fit(A,B,Y)
        self.task = self.make_task(N,D,M, self.ncodebooks, self.lut_work_const)
        self.task.X=A
        self.task.Q=B
        copy_python_to_amm(self, self.task.amm)
        if not self.create_lut_at_test:
            #copy_python_luts(self, self.task.amm)
            self.task.lut()# in C++ has R^2 of 0.999 with Py
    
    def set_B(self,B):
        # called in _eval_task when Keeping Q fixed and only changing X
        super().set_B(B)
        #self.luts, self.offset, self.scale = self.enc.encode_Q(B.T)
        copy_python_to_amm(self, self.task.amm)
        if not self.create_lut_at_test:
            self.task.lut()
            
    #What outputs Y_hat matrix, assume only called once? 
    def __call__(self, A, B):
        if self.task is None:
            self.fit(A,B)
        # True Encodes test Q as LUT instead of using train_Q's luts 
        self.task.run_matmul(self.create_lut_at_test) 
        Y_hat=self.task.amm.out_mat 
        if not self.keep_cpp_type:
            Y_hat=(Y_hat.astype(np.uint16)*self.task.amm.ncodebooks/self.task.amm.out_scale) + self.task.amm.out_offset_sum
        return Y_hat
               