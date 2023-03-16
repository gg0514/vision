# -*- coding: utf-8 -*-
"""
Created on Tue Mar 14 13:03:53 2023

@author: MC-IBD-Dev
"""

import cv2 as cv

img= cv.imread('girl.jpg')
gray= cv.cvtColor(img, cv.COLOR_BGR2GRAY)

sift= cv.SIFT_create()
kp,des= sift.detectAndCompute(gray, None)

gray= cv.drawKeypoints(gray, kp, None, flags=cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

cv.imshow('sift', gray)

cv.waitKey(0)
cv.destroyAllWindows()
