# -*- coding: utf-8 -*-
"""
Created on Tue Mar 14 10:31:56 2023

@author: MC-IBD-Dev
"""

import cv2 as cv

#img= cv.imread('girl.jpg')
img= cv.imread('../yolov3/images/dog.jpg')
gray= cv.cvtColor(img, cv.COLOR_BGR2GRAY)

canny1= cv.Canny(gray, 50,150)
canny2= cv.Canny(gray, 100,200)

cv.imshow('original', gray)
cv.imshow('canny1', canny1)
cv.imshow('canny2', canny2)

cv.waitKey(0)
cv.destroyAllWindows()
