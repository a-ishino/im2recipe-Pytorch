import os
import simplejson as json
from params import get_parser

from tqdm import tqdm

# =============================================================================
import params
parser = params.get_parser()
opts = parser.parse_args()
data_path, out_path = params.show_mk_small_data_opts(opts)
# =============================================================================

def load(name):
    with open(os.path.join(data_path, name + '.json')) as f_layer:
        return json.load(f_layer)
    
def dump(name, data):
    with open(os.path.join(out_path, name + '.json') ,'w') as f:
        json.dump(data, f)

    
### layer1 の partition情報を用いて小さいデータセットを作成
print("Make smaller layer1...")
layer1 = load('layer1')

smallNs = {'train': 10000, 'val': 2000,'test' : 2000}
output = {'train': [], 'val': [],'test':[]}
counts = {'train': 0, 'val': 0,'test':0}
ids = []

for item in tqdm(layer1):
    key = item["partition"]
    
#     if item['id'] in err_ids:  # なぜか layer1.json にしかない id があるらしくて key error を起こすので、除外
#         continue
        
    counts[key] += 1
    if counts[key] <= smallNs[key]:
        output[key].append(item)
        ids.append(item['id'])
            
small_l1 = output['train'] + output['val'] + output['test']
dump('layer1', small_l1)

for key in counts.keys():
    print("{}: {} -> {}".format(key, counts[key], len(output[key])))

    
### ids を用いて layer2 を小さくする
print("Make smaller layer2...")
layer2 = load('layer2')
small_l2 = []
for item in tqdm(layer2):
    if item['id'] in ids:
        small_l2.append(item)
dump('layer2', small_l2)

print("{} -> {}".format(len(layer2), len(small_l2)))

### ids を用いて det_ingrs を小さくする
print("Make smaller det_ingrs...")
det_ingrs = load('det_ingrs')
small_di = []
for item in tqdm(det_ingrs):
    if item['id'] in ids:
        small_di.append(item)
dump('det_ingrs', small_di)

print("{} -> {}".format(len(det_ingrs), len(small_di)))


