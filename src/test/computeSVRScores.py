#!/usr/bin/python2.7
import sys, os
import argparse
import glob
import numpy as np
sys.path.insert(0, '/exports/cyclops/software/ml/libsvm/python/') 
import svmutil

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-r', '--resdir', type=str, required=True,
            help='Results directory')
    parser.add_argument('-m', '--modelfpath', type=str, required=True,
            help='SVR model pickle file path')
    parser.add_argument('-f', '--feature', type=str, default='fc7',
            help='Feature used')

    args = parser.parse_args()
    FEAT_DIR = os.path.join(args.resdir, 'features', args.feature)
    
    imgs = glob.glob(os.path.join(FEAT_DIR, '*'))
    imgs = [os.path.basename(x) for x in imgs]

    cnt = 0
    for img in imgs:
        cnt = cnt + 1
        feats = []
        cur_feat_dir = os.path.join(FEAT_DIR, img)
        OUT_PATH = os.path.join(cur_feat_dir, 'scores.txt')
        LOCK_PATH = OUT_PATH + '.lock'

        if os.path.exists(OUT_PATH) or os.path.exists(LOCK_PATH):
            continue
        mkdir_p(LOCK_PATH)
        print('Doing for %s %d / %d' % (img, cnt, len(imgs)))

        for i in range(1, 101):
            try:
                feats.append(np.fromfile(os.path.join(cur_feat_dir, str(i) + '.txt'), sep='\n'))
            except:
                continue
        feats = np.array(feats)

        model = svmutil.svm_load_model(args.modelfpath)
        nFeat = np.shape(feats)[0]
        labels, acc, _ = svmutil.svm_predict(np.ones((nFeat, 1)), feats.tolist(), model)
        np.savetxt(OUT_PATH, labels, '%.7f') 
        rmdir_noerror(LOCK_PATH)
        print('done')

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
