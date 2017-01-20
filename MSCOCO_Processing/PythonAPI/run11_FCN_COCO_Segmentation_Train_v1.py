#!/usr/bin/python
# -*- coding: utf-8 -*-
__author__ = 'ar'

import os
import sys
import time
import numpy as np
import json

import skimage.io as skio
import skimage.transform as sktf
import skimage.color as skolor
import pandas as pd
import matplotlib.pyplot as plt

try:
   import cPickle as pickle
except:
   import pickle

import keras.optimizers as opt
from keras.utils.visualize_util import plot as kplot

from run10_common_onimage import BatcherOnImageCOCO, buildModelOnImage_COCO, split_list_by_blocks

#######################################
def usage(pargv):
    print ('Usage: %s {/path/to/train-idx.txt} {/path/to/validation-idx.txt}' % pargv[0])

#######################################
if __name__=='__main__':
    if len(sys.argv)>4:
        parNumEpoch  = int(sys.argv[1])
        parOptimizer = sys.argv[2]
        fidxTrn      = sys.argv[3]
        fidxVal      = sys.argv[4]
    else:
        parNumEpoch = 1000
        parOptimizer = 'adam'
        fidxTrn = '/mnt/data1T2/datasets2/mscoco/raw-data/train2014-food2-128x128/idx.txt'
        fidxVal = '/mnt/data1T2/datasets2/mscoco/raw-data/val2014-food2-128x128/idx.txt'
    if not os.path.isfile(fidxTrn):
        usage(sys.argv)
        raise Exception('Cant find Train-Index path: [%s]' % fidxTrn)
    if not os.path.isfile(fidxVal):
        usage(sys.argv)
        raise Exception('Cant find Validation-Index path: [%s]' % fidxVal)
    #
    parBatchSizeTrain = 128
    parBatchSizeVal   = 128
    parIsTheanoShape  = True
    batcherTrain = BatcherOnImageCOCO(pathDataIdx=fidxTrn,
                                      isTheanoShape=parIsTheanoShape)
    batcherVal   = BatcherOnImageCOCO(pathDataIdx=fidxVal,
                                      isTheanoShape=parIsTheanoShape)
    print (':: Train data: %s' % batcherTrain)
    print (':: Val   data: %s' % batcherVal)
    parNumIterPerEpochTrain = batcherTrain.getNumImg() / parBatchSizeTrain
    parNumIterPerEpochVal   = batcherVal.getNumImg() / parBatchSizeTrain
    parInputShape = batcherTrain.shapeImg
    model,_ = buildModelOnImage_COCO(inpShape=parInputShape, numCls=batcherTrain.numCls, isTheanoFrmwk=parIsTheanoShape)
    model.compile(loss='categorical_crossentropy',
                  optimizer=parOptimizer,
                  # optimizer=opt.SGD(lr=0.01, momentum=0.8, nesterov=True),
                  metrics=['accuracy'])
    model.summary()
    # fimgModel = 'ct-segnet-model-tf.jpg'
    # kplot(model, fimgModel, show_shapes=True)
    # plt.imshow(skio.imread(fimgModel))
    # plt.title(batcherTrain.shapeImg)
    # plt.show(block=False)
    #
    t0 = time.time()
    for eei in range(parNumEpoch):
        print ('[TRAIN] Epoch [%d/%d]' % (eei, parNumEpoch))
        # (0) prepare params
        tmpT1 = time.time()
        stepPrintTrain = int(parNumIterPerEpochTrain / 5)
        stepPrintVal   = int(parNumIterPerEpochVal / 5)
        if stepPrintTrain < 1:
            stepPrintTrain = parNumIterPerEpochTrain
        if stepPrintVal < 1:
            stepPrintVal = parNumIterPerEpochVal
        # (1) model train step
        for ii in range(parNumIterPerEpochTrain):
            dataX, dataY = batcherTrain.getBatchData(parBatchSize=parBatchSizeTrain)
            tret = model.train_on_batch(dataX, dataY)
            if (ii % stepPrintTrain) == 0:
                print ('\t[train] epoch [%d/%d], iter = [%d/%d] : loss[iter]/acc[iter] = %0.3f/%0.2f%%'
                       % (eei, parNumEpoch, ii, parNumIterPerEpochTrain, tret[0], 100. * tret[1]))
        tmpDT = time.time() - tmpT1
        print ('\t*** train-time for epoch #%d is %0.2fs' % (eei, tmpDT))
        # (2) model validation model step
        if ((eei + 1) % 5) == 0:
            print ('[VALIDATION] Epoch [%d/%d]' % (eei, parNumEpoch))
            lstRanges = split_list_by_blocks(range(batcherVal.numImg), parBatchSizeVal)
            tmpVal = []
            for ii, ll in enumerate(lstRanges):
                dataX, dataY = batcherVal.getBatchDataByIdx(parBatchIdx=ll)
                tret = model.evaluate(dataX, dataY, verbose=False)
                tmpVal.append(tret)
                if (ii % stepPrintVal) == 0:
                    print ('\t\t[val] epoch [%d/%d], iter = [%d/%d] : loss[iter]/acc[iter] = %0.3f/%0.2f%%'
                           % (eei, parNumEpoch, ii, parNumIterPerEpochVal, tret[0], 100. * tret[1]))
            tmpVal = np.array(tmpVal)
            tmeanValLoss = np.mean(tmpVal[:, 0])
            tmeanValAcc  = np.mean(tmpVal[:, 1])
            print ('\t::validation: mean-losss/mean-acc = %0.3f/%0.3f' % (tmeanValLoss, tmeanValAcc))
        # (3) export model step
        if ((eei + 1) % 5) == 0:
            tmpT1 = time.time()
            tmpFoutModel = batcherTrain.exportModel(model, eei + 1, extInfo='opt.%s' % parOptimizer)
            tmpDT = time.time() - tmpT1
            print ('[EXPORT] Epoch [%d/%d], export to [%s], time is %0.3fs' % (eei, parNumEpoch, tmpFoutModel, tmpDT))
    dt = time.time() - t0
    print ('Time for #%d Epochs is %0.3fs, T/Epoch=%0.3fs' % (parNumEpoch, dt, dt / parNumEpoch))
