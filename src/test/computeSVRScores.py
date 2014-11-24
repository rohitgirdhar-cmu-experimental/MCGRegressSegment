#!/usr/bin/python2.7
import sys, os
import argparse
import glob
import numpy as np
sys.path.insert(0, '/exports/cyclops/software/ml/libsvm/python/') 
import svmutil
import scipy.io
import pdb

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-r', '--resdir', type=str, required=True,
            help='Results directory')
    parser.add_argument('-m', '--modelfpath', type=str, required=True,
            help='SVR model pickle file path')
    parser.add_argument('-f', '--feature', type=str, default='fc7',
            help='Feature used')
    parser.add_argument('-n', '--numfeatures', type=int, required=True,
            help='Number of features for which to compute the scores. Assumes the presense of 1 to n.txt feature files')

    args = parser.parse_args()
    FEAT_DIR = os.path.join(args.resdir, 'features', args.feature)
    
    imgs = glob.glob(os.path.join(FEAT_DIR, '*'))
    imgs = [os.path.basename(x) for x in imgs]

    cnt = 0
    OUT_PATH = os.path.join(FEAT_DIR, 'scores.txt')
    fout = open(OUT_PATH, 'w')
    
    model = svmutil.svm_load_model(args.modelfpath)
    for seg_id in range(1, args.numfeatures):
        img = str(seg_id) + '.txt'
        feats = []

        print('Doing for %d / %d' % (seg_id, args.numfeatures))

        feats.append(np.fromfile(os.path.join(FEAT_DIR, str(seg_id) + '.txt'), sep='\n'))
        
        feats = np.array(feats)

        nFeat = np.shape(feats)[0]
        labels, acc, _ = svmutil.svm_predict(np.ones((nFeat, 1)), feats.tolist(), model)
        fout.write('%.6f\n' % labels[0])
        fout.flush()
        
        print('done')
    fout.close()

def mkdir_p(path):
    try:
        os.makedirs(path)
    except OSError as exc: # Python >2.5
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else: raise

def rmdir_noerror(path):
    try:
        os.rmdir(path)
    except OSError as exc:
        if exc.errno == errno.EEXIST:
            pass
        else:
            raise

if __name__ == '__main__':
    main()
