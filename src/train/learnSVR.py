import numpy as np
import argparse
import sys, os
import pickle
sys.path.insert(0, '/exports/cyclops/software/ml/libsvm/python/')
import svmutil # for libsvm

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-r', '--resdir', type=str, required=True,
            help="Results directory")
    parser.add_argument('-f', '--feature', type=str, required=True,
            help='feature to use to learn')
    args = parser.parse_args()
    SCORES_FPATH = os.path.join(args.resdir, 'scores.txt')
    FEAT_DIR = os.path.join(args.resdir, 'features', args.feature)

    scores = np.fromfile(SCORES_FPATH, sep='\n')
    feats = []
    for i in range(1, len(scores) + 1):
        feats.append(np.fromfile(os.path.join(FEAT_DIR, str(i) + '.txt'), sep='\n'))
    feats = np.array(feats)
    print('Read all features')
    model = svmutil.svm_train(scores.tolist(), feats.tolist(), '-s 4')
    print svmutil.svm_predict(scores.tolist(), feats.tolist(), model)
    svmutil.svm_save_model(os.path.join(args.resdir, 'svr.model'), model)

if __name__ == '__main__':
    main()

