import word2vec
import sys
import os

'''
Usage: python get_vocab.py /path/to/vocab.bin
'''
w2v_file = sys.argv[1]
model = word2vec.load(w2v_file)
out_path = os.path.join(os.path.dirname(w2v_file),'vocab.txt')

vocab =  model.vocab

print("Writing to %s..." % out_path)
with open(out_path,'w') as f:
    f.write("\n".join(vocab))
print("done")

