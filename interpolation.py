# -*- coding: utf-8 -*-
"""
Created on Tue Mar 14 09:29:28 2023

@author: MC-IBD-Dev
"""

import cv2 as cv

img= cv.imread('girl.jpg')
patch= img[350:450,270:370,:]

img= cv.rectangle(img, (270,350), (370,450),(255,0,0), 3)
patch1= cv.resize(patch, dsize=(0,0), fx=5, fy=5, interpolation=cv.INTER_NEAREST)
patch2= cv.resize(patch, dsize=(0,0), fx=5, fy=5, interpolation=cv.INTER_LINEAR)
patch3= cv.resize(patch, dsize=(0,0), fx=5, fy=5, interpolation=cv.INTER_CUBIC)


cv.imshow('original', img)
cv.imshow('resize nearest', patch1)
cv.imshow('resize bilinear', patch2)
cv.imshow('resize bicubic', patch3)
#cv.imshow('patch', patch)


cv.waitKey(0)
cv.destroyAllWindows()