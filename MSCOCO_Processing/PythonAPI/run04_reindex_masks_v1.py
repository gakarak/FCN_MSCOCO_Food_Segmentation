#!/usr/bin/python
# -*- coding: utf-8 -*-
__author__ = 'ar'

import os
import sys
import shutil
import pandas as pd
import matplotlib.pyplot as plt

import numpy as np
import skimage.io as skio
import skimage.transform as sktf

dictFoods = {
    # 'banana' : 52,
    # 'apple' : 53,
    'sandwich' : 54,
    # 'orange' : 55,
    # 'broccoli' : 56,
    # 'carrot' : 57,
    'hot dog' : 58,
    'pizza' : 59,
    # 'donut' : 60,
    'cake' : 61
}

reversedDirFoods = {vv:kk for kk,vv in dictFoods.items()}

listSortedFoodNames=[
    'pizza',
    'cake',
    'sandwich',
    'hot dog',
    # 'donut',
    # 'banana',
    # 'apple',
    # 'orange',
    # 'broccoli',
    # 'carrot'
]

listSortedFoodIds = [dictFoods[xx] for xx in listSortedFoodNames]
dictCOCO2Index = {dictFoods[xx]:(ii+1) for ii,xx in enumerate(listSortedFoodNames)}

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
            procDir = '%s-%dx%d' % (wdir, parSize, parSize)
            numImages = len(listPathImg)
            for ii,pp in enumerate(listPathImg):
                tfnImg = os.path.basename(pp)
                tfnMskOrig = '%s-mskfood.png' % tfnImg
                tfnMskProc = '%s-mskfood-idx.png' % tfnImg
                finpMskOrig = os.path.join(procDir, tfnMskOrig)
                foutMskProc = os.path.join(procDir, tfnMskProc)
                if (ii%200)==0:
                    print ('\t\t[%d/%d] : %s --> %s' % (ii, numImages, tfnMskOrig, foutMskProc))
                tmskOrig = skio.imread(finpMskOrig)
                tmskProc = np.zeros(tmskOrig.shape, dtype=np.uint8)
                for kk,vv in dictCOCO2Index.items():
                    tmskProc[tmskOrig==kk] = vv
                skio.imsave(foutMskProc, tmskProc)
