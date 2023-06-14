import numpy as np 
from operator import attrgetter, itemgetter

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
  #C++ expects 0 padded values out to 16 per BST of split values, on 1 indexed array
  # [0,v1,v2,v2,v3,v3,v3,v3,v4,v4,v4,v4,v4,v4,v4,v4]
  # (nsplits, [1,2,4,8]) 
  
  raw_splitvals_padded=np.array([np.pad(l, (1,0))
            for l in raw_splitvals
            ])
  #Python Computes: (X-offset) * scale; C++ Computes: X*scale + offset; before comparing to these splitvals need to adjust
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
  
  #assert np.all(amm.getCentroids() == c) #shape wrong
  assert np.all(np.ravel(amm.getCentroids()) == np.ravel(c)) 
  assert np.all(amm.getSplitdims() == d)
  #assert np.all(amm.getSplitvals() == v)
  assert np.all(amm.getEncode_scales()==es)
  assert np.all(amm.getEncode_offsets() == eo)
   
  #del py_est  #to confirm pybind doesn't depend on python memory

def copy_python_luts(est, amm):
  """These aren't really hyperparams in that if Q changes 
  may or may not be efficent to re-compute luts.
  Re-creating luts is quick and uses test version. Not much accuracy difference
  Don't expect Q, therefore luts, to change"""
  luts = np.array([np.ravel(est.luts[i], order='C') 
                   for i in range(len(est.luts))],
                  dtype=np.uint8) 
  amm.luts = luts
  