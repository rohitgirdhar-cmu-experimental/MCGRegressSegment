from sklearn.svm import SVR
import numpy as np
import argparse
import sys, os

def main():
    caffe_root = '/exports/cyclops/software/vision/caffe/'
    sys.path.insert(0, caffe_root + 'python')

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
    clf = SVR()
    clf.fit(feats, scores)
    np.save(os.path.join(args.resdir, 'svr.npy'), clf)

if __name__ == '__main__':
    main()

