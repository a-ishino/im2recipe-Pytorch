import os
import json
import re
import utils
import copy
import pickle
from proc import *
from tqdm import tqdm

import nltk
from nltk.corpus import stopwords

# =============================================================================
import params
parser = params.get_parser()
opts = parser.parse_args()
data_path, out_path, vocab_path = params.show_bigrams_opts(opts)
# =============================================================================
create = opts.create_bigrams # true to compute and store bigrams to disk
# false to go through top N bigrams and create annotations

print('Loading dataset.')
dataset = utils.Layer.merge([utils.Layer.L1, utils.Layer.L2, utils.Layer.INGRS], data_path)

if create:
    print("Creating bigrams...")
    titles = []
    for i in range(len(dataset)):
        title = dataset[i]['title']

        if dataset[i]['partition'] == 'train':
            titles.append(title)
    fileinst = open(os.path.join(out_path, 'titles.txt'),'w')
    for t in titles:
        fileinst.write( t + " ");
    fileinst.close()

    f = open(os.path.join(out_path, 'titles.txt'))
    raw = f.read()
    tokens = nltk.word_tokenize(raw)
    tokens = [i.lower() for i in tqdm(tokens)]
    tokens = [i for i in tqdm(tokens) if i not in stopwords.words('english')]
    #Create your bigrams
    bgs = nltk.bigrams(tokens)
    #compute frequency distribution for all the bigrams in the text
    fdist = nltk.FreqDist(bgs)

    pickle.dump(fdist,open(os.path.join(out_path, 'bigrams.pkl'),'wb'))

else:
    N = 2000
    MAX_CLASSES = 1000
    MIN_SAMPLES = opts.tsamples
    n_class = 1
    ind2class = {}
    class_dict = {}

    fbd_chars = ["," , "&" , "(" , ")" , "'", "'s", "!","?","%","*",".",
                 "free","slow","low","old","easy","super","best","-","fresh",
                 "ever","fast","quick","fat","ww","n'","'n","n","make","con",
                 "e","minute","minutes","portabella","de","of","chef","lo",
                 "rachael","poor","man","ii","i","year","new","style"]

    print('Loading ingr vocab.')
    with open(vocab_path) as f_vocab:
        ingr_vocab = {w.rstrip(): i+2 for i, w in enumerate(f_vocab)} # +1 for lua # ???
        ingr_vocab['</i>'] = 1

    # store number of ingredients (compute only once)
    ningrs_list = []
    for i,entry in enumerate(dataset):

        ingr_detections = detect_ingrs(entry, ingr_vocab)
        ningrs = len(ingr_detections)
        ningrs_list.append(ningrs)

    # load bigrams
    fdist = pickle.load(open(os.path.join(out_path, 'bigrams.pkl'),'rb'))
    Nmost = fdist.most_common(N) # 上位N件の取得？

    # check bigrams
    queries = []
    for oc in tqdm(Nmost):

        counts = {'train': 0, 'val': 0,'test':0}

        if oc[0][0] in fbd_chars or oc[0][1] in fbd_chars: # 頻出語・記号との組み合わせは無視？
            continue

        query = oc[0][0] + ' ' + oc[0][1]
        queries.append(query)
        matching_ids = []
        for i,entry in enumerate(dataset):

            ninstrs = len(entry['instructions'])
            imgs = entry.get('images')
            ningrs =ningrs_list[i]
            title = entry['title'].lower()
            id = entry['id']

            if query in title and ninstrs < opts.maxlen and imgs and ningrs<opts.maxlen and ningrs is not 0: # if match, add class to id
                # we only add if previous class was background
                # or if there is no class for the id
                if id in class_dict:
                    if class_dict[id] == 0:
                        class_dict[id] = n_class
                        counts[dataset[i]['partition']] +=1
                        matching_ids.append(id)
                else:
                    class_dict[id] = n_class
                    counts[dataset[i]['partition']] +=1
                    matching_ids.append(id)

            else: # if there's no match
                if not id in class_dict: # add background class unless not empty
                    class_dict[id] = 0 # background class


        if counts['train'] > MIN_SAMPLES and counts['val'] > 0 and counts['test'] > 0:
            ind2class[n_class] = query
            print(n_class, query, counts)
            n_class+=1
        else:
            for id in matching_ids: # reset classes to background
                class_dict[id] = 0

        if n_class > MAX_CLASSES:
            break

    # get food101 categories (if not present)
    food101 = []
    with open(opts.f101_cats,'r') as f_classes:
        for l in f_classes:
            cls = l.lower().rstrip().replace('_', ' ')
            if cls not in queries:
                food101.append(cls)

    for query in food101:
        counts = {'train': 0, 'val': 0,'test':0}
        matching_ids = []
        for i,entry in enumerate(dataset):

            ninstrs = len(entry['instructions'])
            imgs = entry.get('images')
            ningrs =ningrs_list[i]
            title = entry['title'].lower()
            id = entry['id']

            if query in title and ninstrs < opts.maxlen and imgs and ningrs<opts.maxlen and ningrs is not 0: # if match, add class to id
                # we only add if previous class was background
                # or if there is no class for the id
                if id in class_dict:
                    if class_dict[id] == 0:
                        class_dict[id] = n_class
                        counts[dataset[i]['partition']] +=1
                        matching_ids.append(id)
                else:
                    class_dict[id] = n_class
                    counts[dataset[i]['partition']] +=1
                    matching_ids.append(id)

            else: # if there's no match
                if not id in class_dict: # add background class unless not empty
                    class_dict[id] = 0 # background class

        if counts['train'] > MIN_SAMPLES and counts['val'] > 0 and counts['test'] > 0:
            ind2class[n_class] = query
            print(n_class, query, counts)
            n_class+=1
        else:
            for id in matching_ids: # reset classes to background
                class_dict[id] = 0


    ind2class[0] = 'background'
    print(len(ind2class))
    with open(os.path.join(out_path, 'classes.pkl'),'wb') as f:
        pickle.dump(class_dict,f)
        pickle.dump(ind2class,f)
