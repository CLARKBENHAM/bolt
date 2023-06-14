#%% Mithral inside real NN
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

#%%
gpt2 = torch.hub.load('huggingface/transformers', 'modelForCausalLM', 'gpt2')   

#%%