import os
import json
import numpy as np
import pickle
import sys
from tqdm import *
import time

# =============================================================================
import params
parser = params.get_parser()
opts = parser.parse_args()
data_path, out_path = params.show_tokenize_opts(opts)
# =============================================================================

def readfile(filename):
    with open(filename,'r') as f:
        lines = []
        for line in f.readlines():
            lines.append(line.rstrip())
    return lines

def tok(text,ts=False):

    '''
    Usage: tokenized_text = tok(text,token_list)
    If token list is not provided default one will be used instead.
    '''

    if not ts:
        ts = [',','.',';','(',')','?','!','&','%',':','*','"']

    for t in ts:
        text = text.replace(t,' ' + t + ' ')
    return text



if __name__ == "__main__":

    '''
    Generate tokenized text for w2v training
    Words separated with ' '
    Different instructions separated with \t
    Different recipes separated with \n
    '''

#     try:
#         partition = str(sys.argv[1])
#     except:
#         partition = ''

    dets = json.load(open(os.path.join(data_path, 'det_ingrs.json'),'r'))
    layer1 = json.load(open(os.path.join(data_path, 'layer1.json'),'r'))

    idx2ind = {}
    parts = ['train', 'val', 'test']
    ingrs = []
    for i,entry in enumerate(dets):
        idx2ind[entry['id']] = i


    for part in parts:
        t = time.time()
        with open(os.path.join(out_path, 'tokenized_insts_' + part + '.txt'),'w') as f:
            for i,entry in tqdm(enumerate(layer1)):
                '''
                if entry['id'] in dups:
                    continue
                '''
                if not part=='' and not part == entry['partition']:
                    continue
                    
                instrs = entry['instructions']

                allinstrs = ''
                for instr in instrs:
                    instr =  instr['text']
                    allinstrs+=instr + '\t'

                # find corresponding set of detected ingredients
        #         print(entry['id'])
                det_ingrs = dets[idx2ind[entry['id']]]['ingredients']
                valid = dets[idx2ind[entry['id']]]['valid']

                for j,det_ingr in enumerate(det_ingrs):
                    # if detected ingredient matches ingredient text,
                    # means it did not work. We skip
                    if not valid[j]:
                        continue
                    # underscore ingredient

                    det_ingr_undrs = det_ingr['text'].replace(' ','_')
                    ingrs.append(det_ingr_undrs)
                    allinstrs = allinstrs.replace(det_ingr['text'],det_ingr_undrs)

                f.write(allinstrs + '\n')

        print(time.time() - t, 'seconds.')
        print("Number of unique ingredients",len(np.unique(ingrs)))

        with open(os.path.join(out_path, 'tokenized_insts_' + part + '.txt'),'r') as f:
            text = f.read()
            text = tok(text)

        with open(os.path.join(out_path, 'tokenized_insts_' + part + '.txt'),'w') as f:
            f.write(text)