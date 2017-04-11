#!/usr/bin/python
# -*- coding: utf-8 -*-
__author__ = 'ar'

import time
import shutil
import os
import matplotlib.pyplot as plt
import skimage.io as skio
import numpy as np
import keras
from keras.layers import Conv2D, UpSampling2D, \
    Flatten, Activation, Reshape, MaxPooling2D, Input, merge
from keras.models import Model
import keras.losses
import keras.callbacks as kall

from keras.utils.vis_utils import plot_model as kplot
from keras.utils import np_utils
from keras.utils.vis_utils import plot_model

#####################################################
def buildModelFCNN_UpSampling2D(inpShape=(256, 256, 3), numCls=2, kernelSize=3):
    dataInput = Input(shape=inpShape)
    # -------- Encoder --------
    # Conv #1
    conv1 = Conv2D(filters=4, kernel_size=(kernelSize,kernelSize),
                   padding='same', activation='relu')(dataInput)
    pool1 = MaxPooling2D(pool_size=(2,2))(conv1)
    # Conv #2
    conv2 = Conv2D(filters=8, kernel_size=(kernelSize, kernelSize),
                   padding='same', activation='relu')(pool1)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
    # Conv #3
    conv3 = Conv2D(filters=16, kernel_size=(kernelSize, kernelSize),
                   padding='same', activation='relu')(pool2)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
    # Conv #4
    conv4 = Conv2D(filters=32, kernel_size=(kernelSize, kernelSize),
                   padding='same', activation='relu')(pool3)
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)
    # Conv #5
    conv5 = Conv2D(filters=64, kernel_size=(kernelSize, kernelSize),
                   padding='same', activation='relu')(pool4)
    pool5 = MaxPooling2D(pool_size=(2, 2))(conv5)
    #
    # -------- Decoder --------
    # UpConv #1
    upconv1 = Conv2D(filters=64, kernel_size=(kernelSize, kernelSize),
                     padding='same', activation='relu')(pool5)
    up1 = keras.layers.concatenate([UpSampling2D(size=(2, 2))(upconv1),conv5], axis=-1)
    # UpConv #2
    upconv2 = Conv2D(filters=32, kernel_size=(kernelSize, kernelSize),
                     padding='same', activation='relu')(up1)
    up2 = keras.layers.concatenate([UpSampling2D(size=(2, 2))(upconv2), conv4], axis=-1)
    # UpConv #3
    upconv3 = Conv2D(filters=16, kernel_size=(kernelSize, kernelSize),
                     padding='same', activation='relu')(up2)
    up3 = keras.layers.concatenate([UpSampling2D(size=(2, 2))(upconv3), conv3], axis=-1)
    # UpConv #4
    upconv4 = Conv2D(filters=8, kernel_size=(kernelSize, kernelSize),
                     padding='same', activation='relu')(up3)
    up4 = keras.layers.concatenate([UpSampling2D(size=(2, 2))(upconv4), conv2], axis=-1)
    # UpConv #5
    upconv5 = Conv2D(filters=8, kernel_size=(kernelSize, kernelSize),
                     padding='same', activation='relu')(up4)
    up5 = keras.layers.concatenate([UpSampling2D(size=(2, 2))(upconv5), conv1], axis=-1)
    #
    # 1x1 Convolution: emulation of Dense layer
    convCls = Conv2D(filters=2, kernel_size=(1,1), padding='valid', activation='linear')(up5)
    sizeReshape = np.prod(inpShape[:2])
    ret = Reshape([sizeReshape, 2])(convCls)
    ret = Activation('softmax')(ret)
    retModel = Model(dataInput, ret)
    return retModel

def readDataMasked(pidx):
    with open(pidx, 'r') as f:
        wdir = os.path.dirname(pidx)
        lstpath = f.read().splitlines()
        lstpath = [os.path.join(wdir,xx) for xx in lstpath]
        numPath = len(lstpath)
        dataX = None
        dataY = None
        for ii,pp in enumerate(lstpath):
            img4 = skio.imread(pp)
            img = img4[:,:,:3].astype(np.float)
            img -= img.mean()
            img /= img.std()
            msk = (img4[:,:,3]>0).astype(np.float)
            msk = np_utils.to_categorical(msk.reshape(-1), 2)
            # msk = msk.reshape(-1)
            if dataX is None:
                dataX = np.zeros([numPath] + list(img.shape))
                dataY = np.zeros([numPath] + list(msk.shape))
            dataX[ii] = img
            dataY[ii] = msk
            if (ii%100)==0:
                print ('[%d/%d]' % (ii, numPath))
        return (dataX, dataY)

def _getRand():
    return 2. * (np.random.rand() - 0.5)

def train_generator(pidx, batchSize=64, imsize = 256, isRandomize=True):
    wdir = os.path.dirname(pidx)
    with open(pidx, 'r') as f:
        lstpath = f.read().splitlines()
        lstpath = np.array([os.path.join(wdir, xx) for xx in lstpath])
        numPath = len(lstpath)
        rndMean = 0.2
        rndStd  = 0.06
        while True:
            rndIdx = np.random.randint(0,numPath, batchSize).tolist()
            dataX = np.zeros((batchSize, imsize, imsize, 3))
            dataY = np.zeros((batchSize, imsize*imsize,  2))
            for ii, iidx in enumerate(rndIdx):
                img = skio.imread(lstpath[iidx])
                msk = (img[:,:, 3]>0).astype(np.float)
                img = img[:,:,:3].astype(np.float)
                # (v1)
                # img -= img.mean()
                # img /= img.std()
                # msk = np_utils.to_categorical(msk.reshape(-1), 2)
                # dataX[ii] = img
                # dataY[ii] = msk
                # (v2)
                if np.min(img.shape[:2])==imsize:
                    imgCrop = img.copy()
                    mskCrop = msk.copy()
                else:
                    rndR = np.random.randint(0, img.shape[0] - imsize - 1)
                    rndC = np.random.randint(0, img.shape[1] - imsize - 1)
                    imgCrop = img[rndR:rndR + imsize, rndC:rndC + imsize]
                    mskCrop = msk[rndR:rndR + imsize, rndC:rndC + imsize]
                imgCrop -= imgCrop.mean()
                imgCrop /= imgCrop.std()
                if isRandomize:
                    shiftMean = rndMean * _getRand()
                    shiftStd  = rndStd  * _getRand()
                    imgCrop -= shiftMean
                    imgCrop /= 1.0 + shiftStd
                mskCrop = np_utils.to_categorical(mskCrop.reshape(-1), 2)
                dataX[ii] = imgCrop
                dataY[ii] = mskCrop
            yield (dataX, dataY)

############################
if __name__ == '__main__':
    pathModel = 'test_keras_mode_v3.h5'
    if not os.path.isfile(pathModel):
        model = buildModelFCNN_UpSampling2D()
        model.compile(optimizer='adam',
                      loss='categorical_crossentropy',
                      metrics=['accuracy'])
    else:
        pathModelBk = '%s-%s.bk' % (pathModel, time.strftime('%Y.%m.%d-%H.%M.%S'))
        shutil.copy(pathModel, pathModelBk)
        model = keras.models.load_model(pathModel)
    pathModelPlot = '%s-plot.png' % pathModel
    plot_model(model, to_file=pathModelPlot, show_shapes=True)
    # plt.imshow(skio.imread(pathModelPlot))
    # plt.show()
    model.summary()
    # fidxTrn = '/mnt/data1T2/datasets2/z_all_in_one/Portraits_256x256/idx.txt'
    # fidxTrn = '/mnt/data1T2/datasets2/z_all_in_one/mscoco/coco_4x3/idx.txt'
    fidxTrn = '/mnt/data1T2/datasets2/z_all_in_one/mscoco/train_data/idx.txt'
    fidxVal = '/mnt/data1T2/datasets2/z_all_in_one/BVideo_All_256x256/idx.txt'
    #
    numTrn = len(open(fidxTrn,'r').read().splitlines())
    batchSize = 128
    numEpochs = 100
    numIterPerEpoch = numTrn/batchSize
    pathLog = '%s-log.csv' % pathModel
    #
    # trnX, trnY = readDataMasked(fidxTrn)
    valX, valY = readDataMasked(fidxVal)
    # model.fit(x=trnX, y=trnY, batch_size=128, epochs=100, validation_data=(valX, valY))
    model.fit_generator(
        generator=train_generator(fidxTrn, batchSize=128, imsize=256),
        steps_per_epoch=numIterPerEpoch,
        epochs=numEpochs, validation_data=(valX, valY),
    callbacks=[
        kall.ModelCheckpoint(pathModel, verbose=True, save_best_only=True),
        kall.CSVLogger(pathLog, append=True)
    ])
    # model.save(pathModel)
    # print ('-----')
