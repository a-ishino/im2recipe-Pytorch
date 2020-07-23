import os
import datetime
import argparse
import shutil

def get_parser():

    parser = argparse.ArgumentParser(description='tri-joint parameters')
    # general
    parser.add_argument('-seed',           default=1234, type=int)
    parser.add_argument('--no-cuda',       action='store_true')

    # data
    parser.add_argument('-img_path',       default='data/images/')
    parser.add_argument('-data_path',      default='data/')
    parser.add_argument('-workers',        default=0, type=int) # default=30

    # model
    parser.add_argument('-batch_size',     default=160, type=int)
    parser.add_argument('-snapshots',      default='snapshots/',type=str)

    # im2recipe model
    parser.add_argument('-embDim',         default=1024, type=int)
    parser.add_argument('-nRNNs',          default=1, type=int)
    parser.add_argument('-srnnDim',        default=1024, type=int)
    parser.add_argument('-irnnDim',        default=300, type=int)
    parser.add_argument('-imfeatDim',      default=2048, type=int)
    parser.add_argument('-stDim',          default=1024, type=int)
    parser.add_argument('-ingrW2VDim',     default=300, type=int)
    parser.add_argument('-maxSeqlen',      default=20, type=int)
    parser.add_argument('-maxIngrs',       default=20, type=int)
    parser.add_argument('-maxImgs',        default=5, type=int)
    parser.add_argument('-numClasses',     default=1048, type=int)
    parser.add_argument('-preModel',       default='resNet50',type=str)
    parser.add_argument('-semantic_reg',   default=True,type=bool)
    # parser.add_argument('--semantic_reg', default=False,type=bool)

    # training 
    parser.add_argument('-lr',              default=0.0001, type=float)
    parser.add_argument('-momentum',        default=0.9, type=float)
    parser.add_argument('-weight_decay',    default=0, type=float)
    parser.add_argument('-epochs',          default=300, type=int)
    parser.add_argument('-start_epoch',     default=0, type=int)
    parser.add_argument('-ingrW2V',         default='data/vocab.bin',type=str)
    parser.add_argument('-valfreq',         default=10,type=int)  
    parser.add_argument('-patience',        default=1, type=int)
    parser.add_argument('-freeVision',      default=False, type=bool)
    parser.add_argument('-freeRecipe',      default=True, type=bool)
    parser.add_argument('-cos_weight',      default=0.98, type=float)
    parser.add_argument('-cls_weight',      default=0.01, type=float)
    parser.add_argument('-resume',          default='', type=str)

    # test
    parser.add_argument('-results_path',    default='results/', type=str)
    parser.add_argument('-model_path',      default='snapshots/model_e220_v-4.700.pth.tar', type=str)
    parser.add_argument('-test_image_path', default='chicken.jpg', type=str)    

    # MedR / Recall@1 / Recall@5 / Recall@10
    parser.add_argument('-embtype',         default='image', type=str) # [image|recipe] query type
    parser.add_argument('-medr',            default=1000, type=int) 

    # dataset
    parser.add_argument('-maxlen',          default=20, type=int)
    parser.add_argument('-vocab',           default = 'vocab.txt', type=str)
    parser.add_argument('-dataset',         default = '../data/recipe1M/', type=str)
    parser.add_argument('-sthdir',          default = '../data/', type=str)
    
    parser.add_argument('-inst_emb',        default = 'bert', type=str)
    
    parser.add_argument('-tag',             default = '1M')
    parser.add_argument('-remove1m',        default = '1M', type=str)
    parser.add_argument('-n_samples',       default = 0, type=int)
    
    # tensorboard
    parser.add_argument('-logdir',          default = 'logs/')

    return parser

def show_mk_dataset_opts(opts):
    print("===== SHOW OPTIONS =====")
    
    print("# SOURCES")
    print("-tag : %s" % (opts.tag))
    
    data_path = os.path.join(opts.data_path, opts.tag)
    print("-data_path : %s" % (opts.data_path)) 
    print("  read from : %s" % (data_path))
    if not os.path.exists(os.path.join(data_path, 'layers', 'layer1.json')) or \
       not os.path.exists(os.path.join(data_path, 'layers', 'layer2.json')) or \
       not os.path.exists(os.path.join(data_path, 'layers', 'det_ingrs.json')):
        raise ValueError("can't find layers")
   
    if not os.path.exists(os.path.join(data_path, 'vocab.txt')):
        raise ValueError("can't find vocab.txt in %s" % os.path.join(data_path, 'vocab.txt'))
        
    if not os.path.exists(os.path.join(data_path, 'bigrams/classes.pkl')):
        raise ValueError("can't find classes.pkl in %s" % os.path.join(data_path, 'bigrams/classes.pkl'))
        
    print("-remove1m : %s" % (opts.remove1m))
    if not os.path.exists(opts.remove1m):
        raise ValueError("can't find remove1M.txt in  %s" % opts.remove1m)
    
    print("# TARGETS")
    out_path = os.path.join(opts.data_path, opts.tag, 'lmdbs')
    print("  write to : %s" % (out_path))
    if not os.path.exists(out_path):
        os.makedirs(out_path)
    if os.path.isdir(os.path.join(out_path, 'train_lmdb')):
        shutil.rmtree(os.path.join(out_path, 'train_lmdb'))
    if os.path.isdir(os.path.join(out_path, 'val_lmdb')):
        shutil.rmtree(os.path.join(out_path, 'val_lmdb'))
    if os.path.isdir(os.path.join(out_path, 'test_lmdb')):
        shutil.rmtree(os.path.join(out_path, 'test_lmdb'))
    
    print("# PARAMS")
    print("-inst_emb : %s" % str(opts.inst_emb))
    if not opts.inst_emb is 'bert' and not opts.inst_emb is 'st':
        raise ValueError("inst_emb (%s) most be 'bert' or 'st'" % opts.inst_emb)
    
    print("========================")
    
    return data_path, out_path

def show_train_opts(opts):
    print("===== SHOW OPTIONS =====")
    
    print("# SOURCES")
    print("-tag : %s" % (opts.tag))
    
    print("-img_path : %s" % (opts.img_path)) # path to dir in which train, val and test dir exist.
    if not os.path.exists(os.path.join(opts.img_path,'train')) or not os.path.exists(os.path.join(opts.img_path,'val')):
        raise ValueError("invalid --img_path")
    
    data_path = os.path.join(opts.data_path, opts.tag, 'lmdbs')
    print("-data_path : %s" % (opts.data_path)) # path to dir in which train_lmdb, val_lmdb and test_lmdb dir exist.
    print("  read from : %s" % (data_path))
    if not os.path.exists(os.path.join(data_path,'train_lmdb')) or not os.path.exists(os.path.join(data_path,'val_lmdb')):
        raise ValueError("can't find lmdb dir")
    if not os.path.exists(os.path.join(data_path,'train_keys.pkl')) or not os.path.exists(os.path.join(data_path,'val_keys.pkl')):
        raise ValueError("can't find keys.pkl file")
    if not os.path.exists(os.path.join(data_path, '../vocab.bin')):
        raise ValueError("can't find vocab.bin")
    
    print("# TARGETS")
    print("-snapshots : %s" % (opts.snapshots))
    snapshots = os.path.join(opts.snapshots, opts.tag, datetime.datetime.now().strftime("%Y-%m-%d_%H:%M"))
    print("  write to : %s" % (snapshots))
    os.makedirs(snapshots)
        
    logdir = os.path.join(opts.logdir, opts.tag)
    print("-logdir : %s" % (opts.logdir))
    print("  view by $tensorboard --port xxxx --logdir %s" % (logdir))
    
    print("# PARAMS")
    print("-epochs : %d" % (opts.epochs))
    print("-n_samples : %d" % (opts.n_samples))
    print("-valfreq : %d" % (opts.valfreq))
    print("-inst_emb : %s" % str(opts.inst_emb))
    if not opts.inst_emb is 'bert' and not opts.inst_emb is 'st':
        raise ValueError("inst_emb (%s) most be 'bert' or 'st'" % opts.inst_emb)
    
    print("========================")
    
    return data_path, snapshots, logdir

def show_test_opts(opts):
    print("===== SHOW OPTIONS =====")
    
    print("# SOURCES")
    print("-tag : %s" % (opts.tag))
    
    print("-img_path : %s" % (opts.img_path)) # path to dir in which train, val and test dir exist.
    if not os.path.exists(os.path.join(opts.img_path,'test')):
        raise ValueError("invalid -img_path")
    
    data_path = os.path.join(opts.data_path, opts.tag, 'lmdbs')
    print("-data_path : %s" % (opts.data_path)) # path to dir in which train_lmdb, val_lmdb and test_lmdb dir exist.
    print("  read from : %s" % (data_path))
    if not os.path.exists(os.path.join(data_path,'test_lmdb')):
        raise ValueError("can't find lmdb dir")
    if not os.path.exists(os.path.join(data_path,'test_keys.pkl')):
        raise ValueError("can't find keys.pkl file")
    if not os.path.exists(os.path.join(data_path, '../vocab.bin')):
        raise ValueError("can't find vocab.bin")
    
    print("# TARGETS")
    print("-model_path : %s" % (opts.model_path))
    if not os.path.exists(opts.model_path):
        raise ValueError("invalid -model_path")
    
    results_path = os.path.join(opts.results_path, opts.tag, opts.model_path.split('/')[-2]) # result/recipe1M/2020-07-20_10:00:00/
    print("-results_path : %s" % (opts.results_path))
    print("  write to : %s" % (results_path))
    if not os.path.exists(opts.results_path):
        raise ValueError("invalid -results_path")
    if not os.path.exists(results_path):
        os.makedirs(results_path)
        
    logdir = os.path.join(opts.logdir, opts.tag)
    print("-logdir : %s" % (opts.logdir))
    print("  view by $tensorboard --port xxxx --logdir %s" % (logdir))
    
    print("# PARAMS")
    print("-n_samples : %d" % (opts.n_samples))
    
    print("========================")
    
    return data_path, results_path, logdir




