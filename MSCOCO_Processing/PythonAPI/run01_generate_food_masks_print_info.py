#!/usr/bin/python
# -*- coding: utf-8 -*-
__author__ = 'ar'

import os
import glob

import numpy as np
import matplotlib.pyplot as plt
import pycocotools.coco as coco
from pycocotools.coco import COCO
import skimage.io as skio

######################################
def findCatById(listCats, catId):
    for ii in listCats:
        if ii['id'] == catId:
            return ii
    return None

######################################
if __name__ == '__main__':
    # dataDir = '/home/ar/datasets/mscoco'
    dataDir = '/mnt/data1T2/datasets2/mscoco/raw-data'
    dataType = 'train2014'
    # dataType = 'val2014'
    annFile  = '%s/annotations/instances_%s.json' % (dataDir, dataType)
    imgDir = '%s/%s' % (dataDir, dataType)
    if not os.path.isdir(imgDir):
        raise Exception('Cant find directory with MS-COCO images [%s]' % dataDir)
    #
    coco = COCO(annFile)
    #
    listCatsFoodIdx = coco.getCatIds(supNms=['food'])
    for ii, idx in enumerate(listCatsFoodIdx):
        tmpCat = coco.loadCats(ids = idx)[0]
        print ('%d [%d] : %s (%s)' % (ii, idx, tmpCat['name'], tmpCat['supercategory']))
    print ('-------')