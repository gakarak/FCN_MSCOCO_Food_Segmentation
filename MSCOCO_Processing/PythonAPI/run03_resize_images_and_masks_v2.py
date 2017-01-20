#!/usr/bin/python
# -*- coding: utf-8 -*-
__author__ = 'ar'

import os
import sys
import shutil
import pandas as pd

import numpy as np
import skimage.io as skio
import skimage.transform as sktf

######################################
def makeDirIfNotExists(pathToDir, isCleanIfExists=True):
    """
    create directory if directory is absent
    :param pathToDir: path to directory
    :param isCleanIfExists: flag: clean directory if directory exists
    :return: None
    """
    if os.path.isdir(pathToDir) and isCleanIfExists:
        shutil.rmtree(pathToDir)
    if not os.path.isdir(pathToDir):
        os.makedirs(pathToDir)

######################################
if __name__ == '__main__':
    listSizes=[128,256,512]
    listfIdx=['/mnt/data1T2/datasets2/mscoco/raw-data/train2014-food2/idx.txt',
              '/mnt/data1T2/datasets2/mscoco/raw-data/val2014-food2/idx.txt']
    numSizes = len(listSizes)
    numfIdx  = len(listfIdx)
    for ffi, fidx in enumerate(listfIdx):
        wdir = os.path.dirname(fidx)
        dataIdx = pd.read_csv(fidx, header=None)
        listPathImg = [os.path.join(wdir, xx) for xx in dataIdx[0]]
        print ('[%d/%d] : %s' % (ffi,numfIdx, os.path.basename(wdir)))
        for ssi, parSize in enumerate(listSizes):
            print ('\t[%d/%d] generate size %dx%d' % (ssi, numSizes, parSize,parSize))
            #
            outDir = '%s-%dx%d' % (wdir, parSize,parSize)
            makeDirIfNotExists(outDir, isCleanIfExists=True)
            numImages = len(listPathImg)
            for ii,pp in enumerate(listPathImg):
                tfnImg = os.path.basename(pp)
                tfnMsk = '%s-mskfood.png' % tfnImg
                finpMsk = os.path.join(wdir, tfnMsk)
                foutImg = os.path.join(outDir, tfnImg)
                foutMsk = os.path.join(outDir, tfnMsk)
                if (ii%200)==0:
                    print ('\t\t[%d/%d] : %s --> %s' % (ii, numImages, tfnImg, foutImg))
                timg  = skio.imread(pp)
                tmsk  = skio.imread(finpMsk)
                timgr = sktf.resize(timg, (parSize, parSize), preserve_range=True).astype(np.uint8)
                tmskr = sktf.resize(tmsk, (parSize, parSize), order=0, preserve_range=True).astype(np.uint8)
                skio.imsave(foutImg, timgr)
                skio.imsave(foutMsk, tmskr)
