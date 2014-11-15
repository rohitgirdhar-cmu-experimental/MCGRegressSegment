#!/usr/bin/python2.7

import numpy as np
import scipy
import matplotlib.pyplot as plt
import argparse
import os, errno
import sys
import glob

def main():
    caffe_root = '/exports/cyclops/software/vision/caffe/'
    sys.path.insert(0, caffe_root + 'python')

    parser = argparse.ArgumentParser()
    parser.add_argument('-r', '--resdir', type=str, required=True,
            help='Results directory')
    parser.add_argument('-f', '--feature', type=str, default='fc7',
            help='could be prediction/fc7/pool5 etc')
    parser.add_argument('-v', '--voctestpath', type=str, required=True,
            help='VOC test directory (to read images)') 

    args = parser.parse_args()
    VOC_DIR = args.voctestpath
    FEAT = args.feature
    RES_DIR = args.resdir
    SEG_DIR = os.path.join(RES_DIR, 'mcgprops')
    OUT_DIR = os.path.join(RES_DIR, 'features', FEAT)
    IMGS_DIR = os.path.join(VOC_DIR, 'JPEGImages_person')

    if not os.path.exists(OUT_DIR):
        mkdir_p(OUT_DIR)

    import caffe

    # Set the right path to your model definition file, pretrained model weights,
    # and the image you would like to classify.
    MODEL_FILE = '../train/deploy.prototxt'
    mean = np.load(caffe_root + 'python/caffe/imagenet/ilsvrc_2012_mean.npy')
    # convert into image for visualization and processing
    meanImg = mean.swapaxes(0,1).swapaxes(1,2)

    PRETRAINED = os.path.join(caffe_root, 'models/bvlc_reference_caffenet/bvlc_reference_caffenet.caffemodel')

    net = caffe.Classifier(MODEL_FILE, PRETRAINED,
            mean=mean, channel_swap=(2,1,0), raw_scale=255, image_dims=(256, 256))

    net.set_phase_test()
    net.set_mode_cpu()

    imgsList = glob.glob(os.path.join(SEG_DIR, '*'))
    imgsList = [os.path.basename(x) for x in imgsList]

    if not os.path.isdir(OUT_DIR):
        mkdir_p(OUT_DIR)

    count = 0
    for img in imgsList:
        count += 1
        fpath = os.path.join(IMGS_DIR, img + '.jpg')
        cur_segs_dir = os.path.join(SEG_DIR, img)
        cur_out_dir = os.path.join(OUT_DIR, img)

        if os.path.exists(cur_out_dir):
            print('Some other working on/done for %s\n' % fpath)
            continue
        
        mkdir_p(cur_out_dir)
        input_image = caffe.io.load_image(fpath)
        input_image = scipy.misc.imresize(input_image, (256, 256))

        segs = glob.glob(os.path.join(cur_segs_dir, '*.jpg'))
        for segfile in segs:
            segfile = os.path.basename(segfile)
            seg_image = caffe.io.load_image(os.path.join(cur_segs_dir, segfile))
            seg_image = scipy.misc.imresize(seg_image, (256, 256))
            idx = (seg_image == 0)
            temp_image = input_image
            temp_image[idx] = meanImg[idx]

            prediction = net.predict([temp_image])
    
            if FEAT == 'prediction':
                feature = prediction.flat
            else:
                feature = net.blobs[FEAT].data[0]; # Computing only 1 crop, by def is center crop
                feature = feature.flat

            segname, _ = os.path.splitext(segfile)
            np.savetxt(os.path.join(cur_out_dir, segname + '.txt'), feature, '%.7f')
        
        print 'Done for %s (%d / %d)' % (img, count, len(imgslist))

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

