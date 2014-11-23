import numpy as np
import argparse
import sys, os
sys.path.insert(0, '/exports/cyclops/software/ml/libsvm/python/')
import svmutil # for libsvm
import pdb

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
        feats.append(np.fromfile(os.path.join(FEAT_DIR, str(i) + '.txt'), sep='\n').tolist())
#   feats = np.array(feats)
    print('Read all features')
    params = svmutil.svm_parameter('-s 4 -t 2')
    model = svmutil.svm_train(svmutil.svm_problem(scores, feats), params)
    svmutil.svm_save_model(os.path.join(args.resdir, 'svr.model'), model)
    print svmutil.svm_predict(scores, feats, model)

if __name__ == '__main__':
    main()

