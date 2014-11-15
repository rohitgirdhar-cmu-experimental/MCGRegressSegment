#!/usr/bin/python2.7

import numpy as np
import scipy
import matplotlib.pyplot as plt
import argparse
import os, errno
import sys

def main():
    caffe_root = '/exports/cyclops/software/vision/caffe/'
    sys.path.insert(0, caffe_root + 'python')

    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--imagesdir', type=str, required=True,
            help='VOC images to process')
    parser.add_argument('-r', '--resdir', type=str, required=True,
            help='Results directory')
    parser.add_argument('-f', '--feature', type=str, default='prediction',
            help='could be prediction/fc7/pool5 etc')

    args = parser.parse_args()
    IMGS_DIR = args.imagesdir
    FEAT = args.feature
    RES_DIR = args.resdir
    IMGS_LIST_FPATH = os.path.join(RES_DIR, 'topimgs.txt')
    SEG_DIR = os.path.join(RES_DIR, 'top_proposed_masks')
    OUT_DIR = os.path.join(RES_DIR, 'features', FEAT)
    if not os.path.exists(OUT_DIR):
        mkdir_p(OUT_DIR)

    import caffe

    # Set the right path to your model definition file, pretrained model weights,
    # and the image you would like to classify.
    MODEL_FILE = 'deploy.prototxt'
    mean = np.load(caffe_root + 'python/caffe/imagenet/ilsvrc_2012_mean.npy')
    # convert into image for visualization and processing
    meanImg = mean.swapaxes(0,1).swapaxes(1,2)

    PRETRAINED = os.path.join(caffe_root, 'models/bvlc_reference_caffenet/bvlc_reference_caffenet.caffemodel')

    net = caffe.Classifier(MODEL_FILE, PRETRAINED,
            mean=mean, channel_swap=(2,1,0), raw_scale=255, image_dims=(256, 256))

    net.set_phase_test()
    net.set_mode_cpu()
    
    fid = open(IMGS_LIST_FPATH)
    topimgs = fid.readlines()
    topimgs = map(lambda x: x.strip(), topimgs)

    if not os.path.isdir(OUT_DIR):
        mkdir_p(OUT_DIR)

    count = 0
    for topimg in topimgs:
        count += 1
        fpath = os.path.join(IMGS_DIR, topimg + '.jpg')
        segpath = os.path.join(SEG_DIR, str(count) + '.jpg')
        out_fpath = os.path.join(OUT_DIR, str(count) + '.txt')
        lock_fpath = os.path.join(OUT_DIR, str(count) + '.lock')

        if os.path.exists(lock_fpath) or os.path.exists(out_fpath):
            print('Some other working on/done for %s\n' % fpath)
            continue
        
        mkdir_p(lock_fpath)
        input_image = caffe.io.load_image(fpath)
        input_image = scipy.misc.imresize(input_image, (256, 256))
        seg_image = caffe.io.load_image(segpath)
        seg_image = scipy.misc.imresize(seg_image, (256, 256))
        idx = (seg_image == 0)
        input_image[idx] = meanImg[idx]

        prediction = net.predict([input_image])
        if FEAT == 'prediction':
            feature = prediction.flat
        else:
            feature = net.blobs[FEAT].data[0]; # Computing only 1 crop, by def is center crop
            feature = feature.flat

        np.savetxt(out_fpath, feature, '%.7f')
        
        rmdir_noerror(lock_fpath)
        print 'Done for %s (%d / %d)' % (topimg, count, len(topimgs))

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

