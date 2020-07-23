import os
import numpy as np
import torch
from transformers import BertTokenizer, BertModel
import logging
import simplejson as json
from tqdm import tqdm
import pickle

# =============================================================================
import params
parser = params.get_parser()
opts = parser.parse_args()
data_path, results_path, logdir = params.show_bert_opts(opts)
# =============================================================================
if opts.view_emb:
    from torch.utils.tensorboard import SummaryWriter
    pre_writer = SummaryWriter(os.path.join(logdir, "pre"))
# =============================================================================
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

### model の ロード
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased') # Load pre-trained model tokenizer (vocabulary)
model = BertModel.from_pretrained('bert-base-uncased', output_hidden_states = True) # Load pre-trained model (weights)
model.to(device)
model.eval() # Put the model in "evaluation" mode, meaning feed-forward operation.
print("Model loaded")


### json のロード
def load(file):
    with open(file) as f_layer:
        return json.load(f_layer)

### 結果の格納
def dump(obj, file):
    with open(file, 'wb') as f:
        pickle.dump(obj, f)


layer1 = load(os.path.join(data_path, 'layer1.json'))
print("JSON loaded : %d " % len(layer1))

### embedding の獲得
bert_vecs_train = {'ids': [], 'encs': [], 'rlens': [], 'rbps': []}
bert_vecs_val = {'ids': [], 'encs': [], 'rlens': [], 'rbps': []}
bert_vecs_test = {'ids': [], 'encs': [], 'rlens': [], 'rbps': []}
bert_vecs = {'train': bert_vecs_train, 'val': bert_vecs_val, 'test': bert_vecs_test}

if opts.view_emb:
    inst_lists = {'train' : [], 'val' : [], 'test' : []}
    title_lists = {'train' : [], 'val' : [], 'test' : []}

for i, item in tqdm(enumerate(layer1), miniters=10000):
    partition = item['partition']
    bert_vec = bert_vecs[partition]
    
    if opts.view_emb:
        inst_list = inst_lists[partition]
        title_list = title_lists[partition]
    
    bert_vec['ids'].append(item['id'])
    bert_vec['rlens'].append(len(item['instructions']))
#     inst_list.append(''.join([inst['text'] + '\n' for inst in item['instructions']]))
#     title_list.append(item['title'])
    
    if len(bert_vec['rbps']):
        bert_vec['rbps'].append(bert_vec['rbps'][-1] + bert_vec['rlens'][-2])
    else:
        bert_vec['rbps'].append(1)
    
    for j, inst in enumerate(item['instructions']):
        marked_text = "[CLS] " + inst['text'] + " [SEP]"
        tokenized_text = tokenizer.tokenize(marked_text)
        indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)
        segments_ids = [1] * len(tokenized_text)
        
        tokens_tensor = torch.tensor([indexed_tokens]).to(device)
        segments_tensors = torch.tensor([segments_ids]).to(device)
        with torch.no_grad():
            outputs = model(tokens_tensor, segments_tensors)
            hidden_states = outputs[2]
        token_vecs = hidden_states[-2][0]
        sentence_embedding = torch.mean(token_vecs, dim=0).to('cpu').detach().numpy() # 2次元ndarrayとして保存する
        
        bert_vec['encs'].append(sentence_embedding)
        
        if opts.view_emb:
            inst_list.append("%d-%d : %s" % (i, j, inst['text']))
            title_list.append("%d-%d : %s" % (i, j, item['title']))
            
print("embedded")
        
for part in ['train', 'val', 'test']:
    dump(bert_vecs[part], os.path.join(results_path, 'encs_' + part + '_768.pkl'))
    if opts.view_emb:
        pre_writer.add_embedding(np.array(bert_vecs[part]['encs']), metadata=inst_lists[part], tag=part + '-inst')
        pre_writer.add_embedding(np.array(bert_vecs[part]['encs']), metadata=title_lists[part], tag=part + '-title')
print("saved")        