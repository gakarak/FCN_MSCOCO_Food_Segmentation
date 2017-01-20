#!/usr/bin/python
# -*- coding: utf-8 -*-
__author__ = 'ar'

import glob
import os
import sys
import time
import numpy as np
import json


import matplotlib.pyplot as plt

try:
   import cPickle as pickle
except:
   import pickle

import skimage.io as skio
import skimage.transform as sktf
import skimage.color as skolor
import pandas as pd
import matplotlib.pyplot as plt

import keras
from keras import backend as K
from keras.datasets import mnist
from keras.models import Sequential, Model
from keras.layers import Dense, Convolution2D, Activation, MaxPooling2D,\
    Flatten, BatchNormalization, InputLayer, Dropout, Reshape, Permute, Input, UpSampling2D, Lambda
from keras.layers.normalization import BatchNormalization
from keras.utils import np_utils
from keras.utils.visualize_util import plot as kplot

import tensorflow as tf
##############################################
def split_list_by_blocks(lst, psiz):
    tret = [lst[x:x + psiz] for x in range(0, len(lst), psiz)]
    return tret

##############################################
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

######################################################
class BatcherOnImageCOCO:
    pathDataIdx=None
    pathMeanData=None
    meanPrefix='mean.pkl'
    arrPathDataImg=None
    arrPathDataMsk=None
    wdir=None
    dataImg     = None
    dataMsk     = None
    dataMskCls  = None
    meanData = None
    #
    imgScale  = 1.
    modelPrefix = None
    #
    isTheanoShape=True
    isRemoveMeanImage=False
    isDataInMemory=False
    shapeImg = None
    numCh  = 1
    numImg = -1
    numCls = -1
    def __init__(self, pathDataIdx, pathMeanData=None, isRecalculateMeanIfExist=False,
                 isTheanoShape=True,
                 isRemoveMeanImage=False,
                 isLoadIntoMemory=False):
        self.isTheanoShape=isTheanoShape
        self.isRemoveMeanImage=isRemoveMeanImage
        # (1) Check input Image
        if not os.path.isfile(pathDataIdx):
            raise Exception('Cant find input Image file [%s]' % pathDataIdx)
        self.pathDataIdx = os.path.abspath(pathDataIdx)
        self.wdir = os.path.dirname(self.pathDataIdx)
        tdata = pd.read_csv(self.pathDataIdx, header=None)
        # (2) Check input Image Mask
        self.arrPathDataImg = np.array([os.path.join(self.wdir, xx) for xx in tdata[0]])
        self.arrPathDataMsk = np.array(['%s-mskfood-idx.png' % xx for xx in self.arrPathDataImg])
        # (3) Load Image and Mask
        tpathImg = self.arrPathDataImg[0]
        tpathMsk = self.arrPathDataMsk[0]
        if not os.path.isfile(tpathImg):
            raise Exception('Cant find CT Image file [%s]' % tpathImg)
        if not os.path.isfile(tpathMsk):
            raise Exception('Cant find CT Image Mask file [%s]' % tpathMsk)
        tdataImg = skio.imread(tpathImg)
        tdataMsk = skio.imread(tpathMsk)
        tdataImg = self.adjustImage(self.transformImageFromOriginal(tdataImg))
        tdataMsk = self.transformImageFromOriginal(tdataMsk)
        self.numCls = len(listSortedFoodNames)+1
        tdataMskCls = self.convertMskToOneHot(tdataMsk)
        self.shapeImg = tdataImg.shape
        self.shapeMsk = tdataMskCls.shape
        # (5) Load data into memory
        self.numImg = len(self.arrPathDataImg)
        if isLoadIntoMemory:
            self.isDataInMemory = True
            self.dataImg = np.zeros([self.numImg] + list(self.shapeImg), dtype=np.float)
            self.dataMsk = np.zeros([self.numImg] + list(self.shapeImg), dtype=np.float)
            self.dataMskCls = np.zeros([self.numImg] + list(self.shapeMsk), dtype=np.float)
            print (':: Loading data into memory:')
            for ii in range(self.numImg):
                tpathImg = self.arrPathDataImg[ii]
                tpathMsk = self.arrPathDataMsk[ii]
                #
                tdataImg = self.adjustImage(skio.imread(tpathImg))
                tdataMsk = skio.imread(tpathMsk)
                tdataImg = self.transformImageFromOriginal(tdataImg)
                tdataMsk = self.transformImageFromOriginal(tdataMsk)
                tdataMskCls = self.convertMskToOneHot(tdataMsk)
                self.dataImg[ii] = tdataImg
                self.dataMsk[ii] = tdataMsk
                self.dataMskCls[ii] = tdataMskCls
                if (ii % 10) == 0:
                    print ('\t[%d/%d] ...' % (ii, self.numImg))
            print ('\t... [done]')
            if self.isTheanoShape:
                tshp = self.dataMskCls.shape
                print (tshp)
        else:
            self.isDataInMemory = False
            self.dataImg    = None
            self.dataMsk    = None
            self.dataMskCls = None
    def getNumImg(self):
        if self.isInitialized():
            return self.numImg
        else:
            return 0
    def adjustImage(self, pimg):
        tret = (1./255.)*pimg.astype(np.float) - 0.5
        return tret
    def convertMskToOneHot(self, msk):
        tshape = list(msk.shape)
        if self.numCls>2:
            tret = np_utils.to_categorical(msk.reshape(-1), self.numCls)
        else:
            tret = (msk.reshape(-1)>0).astype(np.float)
            tret = np.vstack((1.-tret,tret)).transpose()
        if self.isTheanoShape:
            tmpShape = list(tshape[1:]) + [self.numCls]
            # tshape[ 0] = self.numCls
        else:
            tmpShape = tshape
            tmpShape[-1] = self.numCls
        tret = tret.reshape(tmpShape)
        if self.isTheanoShape:
            #FIXME: work only for 2D!!! (not 3D)
            tret = tret.transpose((2,0,1))
        return tret
    def isInitialized(self):
        return (self.shapeImg is not None) and (self.shapeMsk is not None) and (self.wdir is not None) and (self.numCls>0)
    def checkIsInitialized(self):
        if not self.isInitialized():
            raise Exception('class Batcher() is not correctly initialized')
    def toString(self):
        if self.isInitialized():
            if self.meanData is not None:
                tstr = 'Shape=%s, #Samples=%d, #Labels=%d, meanValuePerCh=%s' % (self.shapeImg, self.numImg, self.numCls, self.meanData['meanCh'])
            else:
                tstr = 'Shape=%s, #Samples=%d, #Labels=%d, meanValuePerCh= is Not Calculated' % (self.shapeImg, self.numImg, self.numCls)
        else:
            tstr = "BatcherOnImage2D() is not initialized"
        return tstr
    def __str__(self):
        return self.toString()
    def __repr__(self):
        return self.toString()
    def preprocImageShape(self, img):
        if self.isTheanoShape:
            if img.ndim==2:
                return img.reshape([1] + list(img.shape))
            else:
                return img.transpose((2,0,1))
        else:
            if img.ndim==2:
                return img.reshape(list(img.shape) + [1])
            else:
                return img
    def transformImageFromOriginal(self, pimg):
        tmp = self.preprocImageShape(pimg)
        return tmp.astype(np.float)
    def getBatchDataByIdx(self, parBatchIdx):
        rndIdx = parBatchIdx
        parBatchSize = len(rndIdx)
        dataX = np.zeros([parBatchSize] + list(self.shapeImg), dtype=np.float)
        dataY = np.zeros([parBatchSize] + list(self.shapeMsk), dtype=np.float)
        for ii, tidx in enumerate(rndIdx):
            if self.isDataInMemory:
                dataX[ii] = self.dataImg[tidx]
                dataY[ii] = self.dataMskCls[tidx]
            else:
                tpathImg = self.arrPathDataImg[tidx]
                tpathMsk = self.arrPathDataMsk[tidx]
                tdataImg = self.adjustImage(skio.imread(tpathImg))
                tdataMsk = skio.imread(tpathMsk)
                tdataImg = self.transformImageFromOriginal(tdataImg)
                tdataMsk = self.transformImageFromOriginal(tdataMsk)
                tdataMskCls = self.convertMskToOneHot(tdataMsk)
                dataX[ii] = tdataImg
                dataY[ii] = tdataMskCls
        if self.isTheanoShape:
            tshp = dataY.shape
            dataY = dataY.reshape([tshp[0], tshp[1], np.prod(tshp[-2:])]).transpose((0, 2, 1))
            # print (tshp)
        return (dataX, dataY)
    def getBatchData(self, parBatchSize=8):
        self.checkIsInitialized()
        numImg = self.numImg
        rndIdx = np.random.permutation(range(numImg))[:parBatchSize]
        return self.getBatchDataByIdx(rndIdx)
    def exportModel(self, model, epochId, extInfo=None):
        if extInfo is not None:
            modelPrefix = extInfo
        else:
            modelPrefix = ''
        foutModel = "%s-e%03d.json" % (modelPrefix, epochId)
        foutWeights = "%s-e%03d.h5" % (modelPrefix, epochId)
        foutModel = '%s-%s' % (self.pathDataIdx, foutModel)
        foutWeights = '%s-%s' % (self.pathDataIdx, foutWeights)
        with open(foutModel, 'w') as f:
            str = json.dumps(json.loads(model.to_json()), indent=3)
            f.write(str)
        model.save_weights(foutWeights, overwrite=True)
        return foutModel
    def buildModel_TF(self, targetImageShaped=None):
        if not self.checkIsInitialized():
            retModel = buildModelOnImageCT_TF(inpShape=self.shapeImg, numCls=self.numCls)
            print ('>>> BatcherOnImage2D::buildModel() with input shape: %s' % list(retModel[0].input_shape) )
            return retModel
        else:
            raise Exception('*** BatcherOnImage2D is not initialized ***')
    @staticmethod
    def loadModelFromJson(pathModelJson):
        if not os.path.isfile(pathModelJson):
            raise Exception('Cant find JSON-file [%s]' % pathModelJson)
        tpathBase = os.path.splitext(pathModelJson)[0]
        tpathModelWeights = '%s.h5' % tpathBase
        if not os.path.isfile(tpathModelWeights):
            raise Exception('Cant find h5-Weights-file [%s]' % tpathModelWeights)
        with open(pathModelJson, 'r') as f:
            tmpStr = f.read()
            model = keras.models.model_from_json(tmpStr)
            model.load_weights(tpathModelWeights)
        return model
    @staticmethod
    def loadModelFromDir(pathDirWithModels, paramFilter=None):
        if paramFilter is None:
            lstModels = glob.glob('%s/*.json' % pathDirWithModels)
        else:
            lstModels = glob.glob('%s/*%s*.json' % (pathDirWithModels, paramFilter))
        pathJson  = os.path.abspath(sorted(lstModels)[-1])
        print (':: found model [%s] in directory [%s]' % (os.path.basename(pathJson), pathDirWithModels))
        return BatcherOnImageCOCO.loadModelFromJson(pathJson)

######################################################
def buildModelOnImage_COCO(inpShape=(128, 128, 64, 1), numCls=2, sizFlt=3, isTheanoFrmwk=True):
    dataInput = Input(shape=inpShape)
    # -------- Encoder --------
    # Conv1
    x = Convolution2D(nb_filter=16, nb_col=sizFlt, nb_row=sizFlt, border_mode='same', activation='relu')(dataInput)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    # Conv2
    x = Convolution2D(nb_filter=32, nb_col=sizFlt, nb_row=sizFlt, border_mode='same', activation='relu')(x)
    x = Convolution2D(nb_filter=32, nb_col=sizFlt, nb_row=sizFlt, border_mode='same', activation='relu')(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    # Conv3
    x = Convolution2D(nb_filter=64, nb_col=sizFlt, nb_row=sizFlt, border_mode='same', activation='relu')(x)
    x = Convolution2D(nb_filter=64, nb_col=sizFlt, nb_row=sizFlt, border_mode='same', activation='relu')(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    # Conv4
    x = Convolution2D(nb_filter=128, nb_col=sizFlt, nb_row=sizFlt, border_mode='same', activation='relu')(x)
    x = Convolution2D(nb_filter=128, nb_col=sizFlt, nb_row=sizFlt, border_mode='same', activation='relu')(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    # Conv5
    # x = Convolution3D(nb_filter=256, nb_col=sizFlt, nb_row=sizFlt, border_mode='same', activation='relu')(x)
    # x = MaxPooling3D(pool_size=(2, 2))(x)
    # Conv6
    # x = Convolution3D(nb_filter=256, nb_col=sizFlt, nb_row=sizFlt, border_mode='same', activation='relu')(x)
    # x = MaxPooling3D(pool_size=(2, 2))(x)
    # -------- Decoder --------
    # UpConv #1
    x = Convolution2D(nb_filter=128, nb_col=sizFlt, nb_row=sizFlt, border_mode='same', activation='relu')(x)
    x = Convolution2D(nb_filter=128, nb_col=sizFlt, nb_row=sizFlt, border_mode='same', activation='relu')(x)
    x = UpSampling2D(size=(2, 2))(x)
    # UpConv #2
    x = Convolution2D(nb_filter=64, nb_col=sizFlt, nb_row=sizFlt, border_mode='same', activation='relu')(x)
    x = Convolution2D(nb_filter=64, nb_col=sizFlt, nb_row=sizFlt, border_mode='same', activation='relu')(x)
    x = UpSampling2D(size=(2, 2))(x)
    # UpConv #3
    x = Convolution2D(nb_filter=32, nb_col=sizFlt, nb_row=sizFlt, border_mode='same', activation='relu')(x)
    x = Convolution2D(nb_filter=32, nb_col=sizFlt, nb_row=sizFlt, border_mode='same', activation='relu')(x)
    x = UpSampling2D(size=(2, 2))(x)
    # UpConv #4
    x = Convolution2D(nb_filter=32, nb_col=sizFlt, nb_row=sizFlt, border_mode='same', activation='relu')(x)
    x = UpSampling2D(size=(2, 2))(x)
    # 1x1 Convolution: emulation of Dense layer
    x = Convolution2D(nb_filter=numCls, nb_col=1, nb_row=1, border_mode='same', activation='linear')(x)
    # -------- Finalize --------
    #
    if isTheanoFrmwk:
        tmpModel = Model(dataInput, x)
        tmpShape = tmpModel.output_shape[-2:]
        sizeReshape = np.prod(tmpShape)
        x = Reshape([numCls, sizeReshape])(x)
        x = Permute((2, 1))(x)
        x = Activation('softmax')(x)
        retModel = Model(dataInput, x)
    else:
        x = Lambda(lambda XX: tf.nn.softmax(XX))(x)
        retModel = Model(dataInput, x)
    retShape = retModel.output_shape[1:-1]
    return (retModel, retShape)

######################################################
if __name__=='__main__':
    fidxTrn = '/mnt/data1T2/datasets2/mscoco/raw-data/train2014-food2-128x128/idx.txt'
    fidxVal = '/mnt/data1T2/datasets2/mscoco/raw-data/val2014-food2-128x128/idx.txt'
    batcherTrn = BatcherOnImageCOCO(pathDataIdx=fidxTrn, isTheanoShape=True)
    batcherVal = BatcherOnImageCOCO(pathDataIdx=fidxVal, isTheanoShape=True)
    print ('Train :      %s' % batcherTrn)
    print ('Validation : %s' % batcherVal)
    #
    dataX, dataY = batcherTrn.getBatchData()
    print ('dataX.shape = %s, dataY.shape = %s' % (list(dataX.shape), list(dataY.shape)))
    #
    model,_ = buildModelOnImage_COCO(inpShape=batcherTrn.shapeImg, numCls=batcherTrn.numCls, isTheanoFrmwk=True)
    model.summary()
