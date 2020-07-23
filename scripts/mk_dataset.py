# !/usr/bin/env python
import os
import random
import pickle
import numpy as np
from proc import *
from tqdm import *
import torchfile
import pickle
import time
import utils
import time
import lmdb

# Maxim number of images we want to use per recipe
maxNumImgs = 5

# =============================================================================
import sys
sys.path.append("..")
import args
parser = args.get_parser()
opts = parser.parse_args()
data_path, out_path = args.show_mk_dataset_opts(opts) # create suffix-added path
# =============================================================================

def get_st(part):
    st_vecs = {}

    if opts.inst_emb == "bert":
        with open(os.path.join(data_path, 'insts', 'encs_' + part + '_768.pkl'), 'rb') as f:
            info = pickle.load(f)
        ids = info['ids']
        imids = []
        for i,id in enumerate(ids):
            imids.append(''.join(i for i in id))
        st_vecs['encs'] = info['encs']   # 特徴量、768次元
        st_vecs['rlens'] = info['rlens'] # instの入力長 (instructionが5文なら5)
        st_vecs['rbps'] = info['rbps']   # 累積の入力数 (これまでの文数の合計)
        st_vecs['ids'] = imids
    elif opts.inst_emb == "st":
        info = torchfile.load(os.path.join(data_path, 'insts', 'encs_' + part + '_1024.t7'))
        ids = info[b'ids']
        imids = []
        for i,id in enumerate(ids):
            imids.append(''.join(chr(i) for i in id))
        st_vecs['encs'] = info[b'encs']
        st_vecs['rlens'] = info[b'rlens']
        st_vecs['rbps'] = info[b'rbps']
        st_vecs['ids'] = imids
    else:
        raise Exception("inst_emb (%s) most be 'bert' or 'st'" % opts.inst_emb)

    print(np.shape(st_vecs['encs']),len(st_vecs['rlens']),len(st_vecs['rbps']),len(st_vecs['ids']))
    return st_vecs


# don't use this file once dataset is clean
with open(opts.remove1m,'r') as f:
    remove_ids = {w.rstrip(): i for i, w in enumerate(f)}

t = time.time()
print ("Loading skip-thought vectors...")

st_vecs_train = get_st('train')
st_vecs_val = get_st('val')
st_vecs_test = get_st('test')

st_vecs = {'train':st_vecs_train,'val':st_vecs_val,'test':st_vecs_test}
stid2idx = {'train':{},'val':{},'test':{}}

for part in ['train','val','test']:
    for i,id in enumerate(st_vecs[part]['ids']):
        stid2idx[part][id] = i

print ("Done.",time.time() - t)

print('Loading dataset.')
# print DATASET
dataset = utils.Layer.merge([utils.Layer.L1, utils.Layer.L2, utils.Layer.INGRS], os.path.join(data_path, 'layers'))
print('Loading ingr vocab.')
with open(os.path.join(data_path, 'vocab.txt')) as f_vocab:
    ingr_vocab = {w.rstrip(): i + 2 for i, w in enumerate(f_vocab)} # +1 for lua ← lua じゃないので +2 を +1 にします。
    ingr_vocab['</i>'] = 1 

with open(os.path.join(data_path, 'bigrams/classes.pkl'),'rb') as f:
    class_dict = pickle.load(f)
    id2class = pickle.load(f)

st_ptr = 0
numfailed = 0

env = {'train' : [], 'val':[], 'test':[]}
env['train'] = lmdb.open(os.path.join(out_path, 'train_lmdb'), map_size=int(1e11))
env['val']   = lmdb.open(os.path.join(out_path, 'val_lmdb'), map_size=int(1e11))
env['test']  = lmdb.open(os.path.join(out_path, 'test_lmdb'), map_size=int(1e11))

print('Assembling dataset.')
img_ids = dict()
keys = {'train' : [], 'val':[], 'test':[]}
for i,entry in tqdm(enumerate(dataset)):

    ninstrs = len(entry['instructions'])
    ingr_detections = detect_ingrs(entry, ingr_vocab)
#     print(ingr_detections)
    ningrs = len(ingr_detections)
    imgs = entry.get('images')

    if ninstrs >= opts.maxlen or ningrs >= opts.maxlen or ningrs == 0 or not imgs or remove_ids.get(entry['id']):
        continue

    ingr_vec = np.zeros((opts.maxlen), dtype='uint16')
    ingr_vec[:ningrs] = ingr_detections 

    partition = entry['partition']

    stpos = stid2idx[partition][entry['id']] #select the sample corresponding to the index in the skip-thoughts data
    beg = st_vecs[partition]['rbps'][stpos] - 1 # minus 1 because it was saved in lua
    end = beg + st_vecs[partition]['rlens'][stpos]

    serialized_sample = pickle.dumps( {'ingrs':ingr_vec, 'intrs':st_vecs[partition]['encs'][beg:end],
        'classes':class_dict[entry['id']]+1, 'imgs':imgs[:maxNumImgs]} ) 

    with env[partition].begin(write=True) as txn:
        txn.put('{}'.format(entry['id']).encode('latin1'), serialized_sample)
    # keys to be saved in a pickle file    
    keys[partition].append(entry['id'])

for k in keys.keys():
    with open(os.path.join(out_path, '{}_keys.pkl'.format(k)),'wb') as f:
        pickle.dump(keys[k],f)

print('Training samples: %d - Validation samples: %d - Testing samples: %d' % (len(keys['train']),len(keys['val']),len(keys['test'])))

