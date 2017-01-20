#!/usr/bin/python
# -*- coding: utf-8 -*-
__author__ = 'ar'

import os
import sys
import time
import numpy as np
import json

import nibabel as nib

import skimage.io as skio
import skimage.transform as sktf
import skimage.color as skolor
import pandas as pd
import matplotlib.pyplot as plt

try:
   import cPickle as pickle
except:
   import pickle

import keras
from keras.utils.visualize_util import plot as kplot

from run10_common_onimage import BatcherOnImageCOCO, buildModelOnImage_COCO, split_list_by_blocks

######################################################
if __name__=='__main__':
    print (':: local directory: %s' % os.getcwd())
    #
    fidxTrn = '/mnt/data1T2/datasets2/mscoco/raw-data/train2014-food2-128x128/idx.txt'
    fidxVal = '/mnt/data1T2/datasets2/mscoco/raw-data/val2014-food2-128x128/idx.txt'
    parIsTheanoShape = True
    parIsTheanoShape = True
    parBatchSizeVal  = 128
    batcherTrain = BatcherOnImageCOCO(pathDataIdx=fidxTrn,
                                      isTheanoShape=parIsTheanoShape)
    batcherVal = BatcherOnImageCOCO(pathDataIdx=fidxVal,
                                    pathMeanData=batcherTrain.pathMeanData,
                                    isTheanoShape=parIsTheanoShape)
    print (':: Train data: %s' % batcherTrain)
    print (':: Val   data: %s' % batcherVal)
    #
    wdir = os.path.dirname(fidxTrn)
    modelTrained = BatcherOnImageCOCO.loadModelFromDir(wdir, paramFilter='adam')
    modelTrained.summary()
    #
    lstIdxSplit = split_list_by_blocks(range(batcherVal.numImg), parBatchSizeVal)
    numSplit = len(lstIdxSplit)
    arrIoU = []
    for ii,ll in enumerate(lstIdxSplit):
        if (ii%20)==0:
            print ('[%d/%d] process batch size [%d]' % (ii, numSplit, len(ll)))
        dataX, dataY = batcherVal.getBatchDataByIdx(parBatchIdx=ll)
        tret = modelTrained.predict_on_batch(dataX)
        # convert to 2D-data
        if batcherVal.isTheanoShape:
            tshape2D = list(dataX.shape[2:])
            tret2D   = tret.reshape([dataX.shape[0]] + tshape2D + [tret.shape[-1]])
            dataY2D  = dataY.reshape([dataX.shape[0]] + tshape2D + [tret.shape[-1]])
        else:
            tret2D = tret
            dataY2D = dataY
        sizSplit = len(ll)
        for iidx,idx in enumerate(ll):
            tpathImg = batcherVal.arrPathDataImg[idx]
            tpathMsk = batcherVal.arrPathDataMsk[idx]
            tmskGTIdx  = np.argsort(-dataY2D[iidx], axis=2)[:, :, 0]
            tmskClsIdx = np.argsort(-tret2D[iidx], axis=2)[:,:,0]

            tnumCls = batcherTrain.numCls
            timgOrig = skio.imread(tpathImg)
            timg     = skio.imread(tpathImg)
            tmskClsIdxTmp  = (128 + 128.*tmskClsIdx/tnumCls).astype(np.uint8)
            tmskClsIdx[tmskClsIdx>0] = tmskClsIdxTmp[tmskClsIdx>0]

            timgR = timg[:, :, 0]
            timgG = timg[:, :, 1]
            timgB = timg[:, :, 2]

            timgR[tmskClsIdx > 0] = tmskClsIdx[tmskClsIdx > 0]
            timgG[tmskClsIdx > 0] = tmskClsIdx[tmskClsIdx > 0]

            # timg[:, :, 0] = timgR
            timg[:, :, 1] = timgG
            # timg[:, :, 2] = timgB
            tmsk = skio.imread(tpathMsk)
            tmskIntersect = np.dstack([tmskClsIdx>0, tmskGTIdx>0, tmskClsIdx>0])
            # Calculate IoU
            tmskfReal = (tmskGTIdx  > 0).reshape(-1)
            tmskfClsf = (tmskClsIdx > 0).reshape(-1)
            tIoU = float(np.sum(tmskfReal&tmskfClsf))/np.sum(tmskfReal|tmskfClsf)
            arrIoU.append(tIoU)
            #
            timgOut=np.hstack([timgOrig, timg, (255*tmskIntersect).astype(np.uint8)])

            foutMskCls = '%s-prv-segm1.jpg' % tpathMsk
            # tsegm.to_filename(foutMsk)
            # skio.imsave(foutMskCls, timgOut)
            print ('\t[%d/%d] * processing : %s --> %s' % (iidx, sizSplit, os.path.basename(tpathMsk), os.path.basename(foutMskCls)))
    arrIoU = np.array(arrIoU)
    meanIoU = arrIoU.mean()
    print ('Mean IoU (Validation) = %0.3f' % meanIoU)
    plt.plot(np.sort(arrIoU))
    plt.grid(True)
    plt.title('<IoU> = %0.3f' % meanIoU)
    plt.show()
