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

from keras.utils import np_utils
from keras.utils.vis_utils import plot_model

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

if __name__ == '__main__':
    pathModel = 'test_keras_mode_v3.h5'
    model = keras.models.load_model(pathModel)
    model.summary()
    # fidxTrn = '/mnt/data1T2/datasets2/z_all_in_one/Portraits_256x256/idx.txt'
    fidxTrn = '/mnt/data1T2/datasets2/z_all_in_one/mscoco/train_data/idx.txt'
    fidxVal = '/mnt/data1T2/datasets2/z_all_in_one/BVideo_All_256x256/idx.txt'
    # trnX, trnY = readDataMasked(fidxTrn)
    valX, valY = readDataMasked(fidxVal)
    # model.fit(x=trnX, y=trnY, batch_size=128, epochs=100, validation_data=(valX, valY))
    # model.save(pathModel)
    retY = model.predict(valX, batch_size=128)
    arrAcc = []
    numVal = valX.shape[0]
    for ii in range(numVal):
        arrAcc.append(model.evaluate(valX[[ii]], valY[[ii]], verbose=False))
        if (ii%20)==0:
            print (':: eval [%d/%d]' % (ii, numVal))
    arrAcc = np.array(arrAcc)
    valY = valY.reshape([valY.shape[0]] + list(valX.shape[1:3]) + [valY.shape[-1]])
    retY = retY.reshape([retY.shape[0]] + list(valX.shape[1:3]) + [retY.shape[-1]])
    idxSorted = np.argsort(arrAcc[:,1])
    plt.subplot(1,2,1)
    plt.plot(sorted(arrAcc[:, 0]))
    plt.title('Loss: %s' % np.mean(arrAcc[:,0]))
    plt.subplot(1,2,2)
    plt.plot(sorted(arrAcc[:, 1]))
    plt.title('Acc: %s' % np.mean(arrAcc[:, 1]))
    plt.grid(True)
    plt.show()
    for ii in idxSorted.tolist():
        plt.subplot(1,2,1)
        plt.imshow(np.mean(valX[ii], axis=2))
        plt.subplot(1,2,2)
        q1 = valY[ii][:,:,1].copy()>0.5
        q2 = retY[ii][:, :, 1].copy()>0.5
        plt.imshow(np.dstack([q1,q2,q1]))
        plt.title('Loss: %0.3f, Acc: %0.3f' % (arrAcc[ii,0], arrAcc[ii,1]))
        plt.show()
    print ('-----')
