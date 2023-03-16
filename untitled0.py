# -*- coding: utf-8 -*-
"""
Created on Mon Mar 13 15:40:59 2023

@author: MC-IBD-Dev
"""

import cv2 as cv
import sys

img= cv.imread('girl.jpg')

if img is None:
    sys.exit('파일을 찾을 수 없습니다')

cv.imshow("img", img)

cv.waitKey(1000)
cv.destoryAllWindows()
