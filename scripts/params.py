import os
import argparse

def get_parser():

    parser = argparse.ArgumentParser(description='tri-joint parameters')

    parser.add_argument('-partition',   dest='partition',       default = 'test')
    parser.add_argument('-nlosscurves', dest='nlosscurves',     default = 3,        type=int)
    parser.add_argument('-embedding',   dest='embedding',       default = 'image')
    parser.add_argument('-medr',        dest='medr',            default = 1000,     type=int)
    parser.add_argument('-tsamples',    dest='tsamples',        default = 20,       type=int)
    parser.add_argument('-maxlen',      dest='maxlen',          default = 20,       type=int)
    parser.add_argument('-maxims',      dest='maxims',          default = 5,        type=int)
    parser.add_argument('-seed',        dest='seed',            default = 42,       type=int)
    parser.add_argument('-imsize',      dest='imsize',          default = 256,      type=int)
    parser.add_argument('-dispfreq',    dest='dispfreq',        default = 1000,     type=int)
    parser.add_argument('-valfreq',     dest='valfreq',         default = 10000,    type=int)
    parser.add_argument('-test_feats',  dest='test_feats',      default = '../results/')

    # new dataset 1M
    parser.add_argument('-f101_cats',   dest='f101_cats',       default = '/groups1/gcb50373/dataset/recipe1M/food101_classes_renamed.txt')
    parser.add_argument('-vocab_path',  dest='vocab_path',      default = '')
    parser.add_argument('-stvecs',      dest='stvecs',          default = '../data/text/')
    parser.add_argument('-dataset',     dest='dataset',         default = '../data/recipe1M/')
    parser.add_argument('-suffix',      dest='suffix',          default = '1M')
    parser.add_argument('-h5_data',     dest='h5_data',         default = '../data/data.h5')
    parser.add_argument('-logfile',     dest='logfile',         default = '')
    parser.add_argument('--nocrtbgrs',  dest='create_bigrams',  action='store_false')
    parser.add_argument('--crtbgrs',    dest='create_bigrams',  action='store_true')
    
    parser.add_argument('-data_path',   dest='data_path',       default = '/groups1/gcb50373/dataset/recipe1M')
    parser.add_argument('-out_path',    dest='out_path',        default = '')
    
    parser.add_argument('-tag',        dest='tag',            default = 'bert')
    parser.add_argument('-logdir',     dest='logdir',         default = '../logs')
    parser.add_argument('--view_emb',  dest='view_emb',       action='store_true')

    parser.set_defaults(create_bigrams=False)


    return parser

def show_mk_small_data_opts(opts):
    print("===== SHOW OPTIONS =====")
    
    print("# SOURCES")
    print("-tag : %s" % (opts.tag))
    
    data_path = opts.data_path
    print("-data_path : %s" % (opts.data_path))
    print("  read from : %s" % (data_path))
    if not os.path.exists(os.path.join(data_path,'layer1.json')) or \
       not os.path.exists(os.path.join(data_path,'layer2.json')) or \
       not os.path.exists(os.path.join(data_path,'det_ingrs.json')):
        raise ValueError("can't find layers")
    
    print("\n# TARGETS")
    out_path = os.path.join(opts.out_path, opts.tag, 'layers')
    print("-out_path : %s" % (opts.out_path))
    print("  write to : %s" % (out_path))
    if not os.path.exists(out_path):
        os.makedirs(out_path)
    
    print("========================")
    
    return data_path, out_path

def show_tokenize_opts(opts):
    print("===== SHOW OPTIONS =====")
    
    print("# SOURCES")
    print("-tag : %s" % (opts.tag))
    
    data_path = os.path.join(opts.data_path, opts.tag, 'layers')
    print("-data_path : %s" % (opts.data_path))
    print("  read from : %s" % (data_path))
    if not os.path.exists(os.path.join(data_path,'layer1.json')) or \
       not os.path.exists(os.path.join(data_path,'det_ingrs.json')):
        raise ValueError("can't find layers")
    
    print("\n# TARGETS")
    out_path = opts.out_path if opts.out_path else os.path.join(opts.data_path, opts.tag, 'insts') 
    print("  write to : %s" % (out_path))
    if not os.path.exists(out_path):
        os.makedirs(out_path)
    
    print("========================")
    
    return data_path, out_path

def show_bigrams_opts(opts):
    print("===== SHOW OPTIONS =====")
    
    print("# SOURCES")
    print("-tag : %s" % (opts.tag))
    
    data_path = os.path.join(opts.data_path, opts.tag, 'layers')
    print("-data_path : %s" % (opts.data_path))
    print("  read from : %s" % (data_path))
    if not os.path.exists(os.path.join(data_path,'layer1.json')) or \
       not os.path.exists(os.path.join(data_path,'layer2.json')) or \
       not os.path.exists(os.path.join(data_path,'det_ingrs.json')):
        raise ValueError("can't find layers")

    print("--crtbgrs : %s" % str(opts.create_bigrams))
    if opts.create_bigrams:
        vocab_path = None
    else:
        vocab_path = opts.vocab_path if opts.vocab_path else os.path.join(opts.data_path, opts.tag, 'vocab.txt')
        print("-vocab_path : %s" % (opts.vocab_path))
        if not os.path.exists(vocab_path):
            raise ValueError("can't find vocab.txt")
    
    print("\n# TARGETS")
    out_path = opts.out_path if opts.out_path else os.path.join(opts.data_path, opts.tag, 'bigrams')
    print("  write to : %s" % (out_path))
    if not os.path.exists(out_path):
        os.makedirs(out_path)
    
    print("========================")
    
    return data_path, out_path, vocab_path

def show_bert_opts(opts):
    print("===== SHOW OPTIONS =====")
    
    print("# SOURCES")
    print("-tag : %s" % (opts.tag))
    
    data_path = os.path.join(opts.data_path, opts.tag, 'layers')
    print("-data_path : %s" % (opts.data_path)) 
    print("  read from : %s" % (data_path))
    if not os.path.exists(os.path.join(data_path,'layer1.json')):
        raise ValueError("can't find layer1.json")
    
    print("# TARGETS")
    results_path = os.path.join(opts.data_path, opts.tag, 'insts')
    print("  write to : %s" % (results_path))
    if not os.path.exists(results_path):
        os.makedirs(results_path)
     
    print("--view_emb : %s" % str(opts.view_emb))
    if opts.view_emb:
        logdir = os.path.join(opts.logdir, opts.tag)
        print("-logdir : %s" % (opts.logdir))
        print("  view embeds by $tensorboard --port xxxx --logdir %s" % (logdir))
    else:
        logdir = None
    
    print("========================")
    
    return data_path, results_path, logdir
