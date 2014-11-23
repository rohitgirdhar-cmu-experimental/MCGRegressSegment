#!/usr/bin/python2.7

import numpy as np
import scipy, scipy.io
import matplotlib.pyplot as plt
import argparse
import os, errno
import sys
import gc
import pdb  # for debugging

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-r', '--resdir', type=str, required=True,
            help='Results directory')

    args = parser.parse_args()
    RES_DIR = args.resdir
    SEL_MAT_PATH = os.path.join(RES_DIR, 'selProposals.mat')
    SCORE_PATH = os.path.join(RES_DIR, 'scores.txt')

    sel = scipy.io.loadmat(SEL_MAT_PATH)
    gc.collect()
    scores = sel['scores'][0]
    f = open(SCORE_PATH, 'w')
    pdb.set_trace()
    for i in range(np.shape(scores)[0]):
        f.write(str(scores[i]) + '\n')
    f.close()

if __name__ == '__main__':
    main()

