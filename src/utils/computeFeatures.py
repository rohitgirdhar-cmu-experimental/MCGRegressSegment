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
    OUT_DIR = os.path.join(RES_DIR, 'features', FEAT)
    SEL_MAT_PATH = os.path.join(RES_DIR, 'selProposals.mat')
    if not os.path.exists(OUT_DIR):
        mkdir_p(OUT_DIR)

    sel = scipy.io.loadmat(SEL_MAT_PATH)
    gc.collect() # required, loadmat is crazy with memory usage
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
    
    nImgs = np.shape(sel['imgs'][0])[0]
    topimgs = []
    imgslist = sel['imgs'][0]
    for i in range(nImgs):
        topimgs.append(imgslist[i][0])

    if not os.path.isdir(OUT_DIR):
        mkdir_p(OUT_DIR)

    count = 0
    for topimg in topimgs:
        count += 1
        fpath = os.path.join(IMGS_DIR, topimg + '.jpg')
        out_fpath = os.path.join(OUT_DIR, str(count) + '.txt')
        lock_fpath = os.path.join(OUT_DIR, str(count) + '.lock')

        if os.path.exists(lock_fpath) or os.path.exists(out_fpath):
            print('Some other working on/done for %s\n' % fpath)
            continue
        
        mkdir_p(lock_fpath)
        input_image = caffe.io.load_image(fpath)
        seg_image = sel['masks'][0][count - 1]
        idx = (seg_image == 0)
        bbox = sel['bboxes'][count - 1]
        input_image_crop = input_image[bbox[0] : bbox[2], bbox[1] : bbox[3]]
        idx = idx[bbox[0] : bbox[2], bbox[1] : bbox[3]]

# WITHOUT MASKING!!!
#        mean_temp = scipy.misc.imresize(meanImg, np.shape(idx))
#        input_image_crop[idx] = meanImg[idx]
#        input_image_final = scipy.misc.imresize(input_image_crop, (256, 256))
        try:
            input_image_res = scipy.misc.imresize(input_image_crop, (256, 256))
            prediction = net.predict([input_image_res])
        except:
            print 'Unable to do for', topimg
            rmdir_noerror(lock_fpath)
            np.savetxt(out_fpath, [])
            continue
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

