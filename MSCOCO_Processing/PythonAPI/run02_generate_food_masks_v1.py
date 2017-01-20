#!/usr/bin/python
# -*- coding: utf-8 -*-
__author__ = 'ar'

import os
import glob

import shutil
import numpy as np
import matplotlib.pyplot as plt
import pycocotools.coco as coco
from pycocotools.coco import COCO
from pycocotools import mask
import skimage.io as skio
import skimage.color as skcl
import warnings

######################################
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
    # dataDir = '/home/ar/datasets/mscoco'
    dataDir = '/mnt/data1T2/datasets2/mscoco/raw-data'
    # dataType = 'train2014'
    dataType = 'val2014'
    dirImg = '%s/%s' % (dataDir, dataType)
    if not os.path.isdir(dirImg):
        raise Exception('Cant find directory with images [%s]' % dirImg)
    dirOut = '%s/%s-food2' % (dataDir, dataType)
    makeDirIfNotExists(pathToDir=dirOut, isCleanIfExists=False)
    #
    annFile  = '%s/annotations/instances_%s.json' % (dataDir, dataType)
    imgDir = '%s/%s' % (dataDir, dataType)
    if not os.path.isdir(imgDir):
        raise Exception('Cant find directory with MS-COCO images [%s]' % dataDir)
    #
    coco = COCO(annFile)
    #
    listCatsFoodIdx = coco.getCatIds(supNms=['food'])
    assert (set(listCatsFoodIdx) == set(listSortedFoodIds))
    for ii, idx in enumerate(listCatsFoodIdx):
        tmpCat = coco.loadCats(ids = idx)[0]
        print ('%d [%d] : %s (%s)' % (ii, idx, tmpCat['name'], tmpCat['supercategory']))
    #
    tmpDictFoodImgIds = {}
    for ii, idx in enumerate(listSortedFoodIds):
        tmpImgIds = coco.getImgIds(catIds=idx)
        for timgId in tmpImgIds:
            if tmpDictFoodImgIds.has_key(timgId):
                tmpDictFoodImgIds[timgId].append(idx)
            else:
                tmpDictFoodImgIds[timgId] = [idx]
    setAllFoodImgIds = sorted(tmpDictFoodImgIds.keys())
    print ('#list/#set = %d' % len(tmpDictFoodImgIds.keys()))
    numImages = len(setAllFoodImgIds)
    for ii, kk in enumerate(setAllFoodImgIds):
        print ('[%d/%d]' % (ii, numImages))
        timgInfo = coco.loadImgs(kk)[0]
        foutImg = '%s/%s' % (dirOut, timgInfo['file_name'])
        foutMsk = '%s/%s-mskfood.png' % (dirOut, timgInfo['file_name'])
        foutPrv = '%s/%s-preview.jpg' % (dirOut, timgInfo['file_name'])
        if os.path.isfile(foutPrv):
            print ('\timage already processed [%s]' % foutImg)
            continue
        #
        fimg = '%s/%s' % (dirImg, timgInfo['file_name'])
        timg = skio.imread(fimg)
        # assert (timg.ndim==3)
        if timg.ndim==2:
            timg = skcl.gray2rgb(timg)
        twidth  = timgInfo['width']
        theight = timgInfo['height']
        vv = tmpDictFoodImgIds[kk]
        tmsk = None
        for vvi in vv:
            tannIds = coco.getAnnIds(imgIds=kk, catIds=vvi, iscrowd=False)
            tanns = coco.loadAnns(tannIds)
            print ('\t :: processing: %d -> %s' % (vvi, reversedDirFoods[vvi]))
            tmpMsk = None
            for ttt in tanns:
                rle = mask.frPyObjects(ttt['segmentation'], theight, twidth)
                tmpm = mask.decode(rle)
                if tmpm.shape[2]>1:
                    print ('\t\t**** multiple shape out :(  --> [%s]' % fimg)
                tmpm = np.sum(tmpm, axis=2)
                if tmpMsk is None:
                    tmpMsk = tmpm
                else:
                    tmpMsk += tmpm
            if tmsk is None:
                tmsk = np.zeros(tmpMsk.shape, dtype=tmpMsk.dtype)
            tmsk[tmpMsk>0]=vvi
        #
        timgPreview = timg.copy()
        chIdx = 1
        timgPreviewCh = timgPreview[:,:,chIdx]
        timgPreviewCh[tmsk>0] = 255
        timgPreview[:,:,chIdx] = timgPreviewCh
        #
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            skio.imsave(foutImg, timg)
            skio.imsave(foutMsk, tmsk[:,:])
            skio.imsave(foutPrv, timgPreview)
    print ('-------')
