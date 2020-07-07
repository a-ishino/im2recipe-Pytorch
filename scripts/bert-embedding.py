import torch
from transformers import BertTokenizer, BertModel
import logging
import simplejson as json
from tqdm import tqdm
import pickle


DATASET = '../data/test/'
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


layer1 = load(DATASET + 'layer1.json')
print("JSON loaded")

### embedding の獲得
bert_vecs_train = {'ids': [], 'encs': [], 'rlens': [], 'rbps': []}
bert_vecs_val = {'ids': [], 'encs': [], 'rlens': [], 'rbps': []}
bert_vecs_test = {'ids': [], 'encs': [], 'rlens': [], 'rbps': []}
bert_vecs = {'train': bert_vecs_train, 'val': bert_vecs_val, 'test': bert_vecs_test}

for item in tqdm(layer1):
    partition = item['partition']
    bert_vec = bert_vecs[partition]
    bert_vec['ids'].append(item['id'])
    bert_vec['rlens'].append(len(item['instructions']))
    
    if len(bert_vec['rbps']):
        bert_vec['rbps'].append(bert_vec['rbps'][-1] + bert_vec['rlens'][-2])
    else:
        bert_vec['rbps'].append(1)
    
    for inst in item['instructions']:
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
        sentence_embedding = torch.mean(token_vecs, dim=0)
        
        bert_vec['encs'].append(sentence_embedding)
print("embedded")
        
for part in ['train', 'val', 'test']:
    dump(bert_vecs[part], DATASET + 'encs_' + part + '_768.pkl')
print("saved")        