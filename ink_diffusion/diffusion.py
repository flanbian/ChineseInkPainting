#!/usr/bin/python
# -*- coding: utf-8 -*-

import cv2
import numpy as np
import random

from skimage import io, color
from random import randint

class Diffusion:
    def __init__(self, image):
    	  self.img = image
        
    def normalize_range(self, image, begin=0, end=255):
        imgSize = image.shape
        imgWidth = imgSize[1]
        imgHeight = imgSize[0]
        
        oldBegin = 0
        oldEnd = 1
        oldRange = oldEnd - oldBegin
        newRange = end - begin

        for height in range(imgHeight):
            for width in range(imgWidth):
                scale = (image.item(height, width) - oldBegin) / oldRange
                image.itemset((height, width), (newRange * scale) + begin)

        return image
        
    def diffusion(self, r, n):
    	  #image = color.rgb2gray(image)
        tmpImg = np.zeros((self.img.shape[0], self.img.shape[1], 3), np.uint8)
        imgSize = self.img.shape
        imgWidth = imgSize[1]
        imgHeight = imgSize[0]
        operationVal = [-1, 1]
        for height in xrange(imgHeight):
            for width in xrange(imgWidth):
                xRand = randint(0, r)
                yRand = randint(0, r - xRand)
                operationX = random.choice(operationVal)
                operationY = random.choice(operationVal)
                if width + (operationX * xRand) < 0:
                    x = width + (-operationX * xRand)
                elif width + (operationX * xRand) > (imgWidth - 1):
                    x = width + (-operationX * xRand)
                else:
                    x = width + (operationX * xRand)
                if height + (operationY * yRand) < 0:
                    y = height + (-operationY * yRand)
                elif height + (operationY * yRand) > (imgHeight - 1):
                    y = height + (-operationY * yRand)
                else:
                    y = height + (operationY * yRand)

                tmpImg.itemset((height, width, 0), self.img.item(y, x, 0))
                tmpImg.itemset((height, width, 1), self.img.item(y, x, 1))
                tmpImg.itemset((height, width, 2), self.img.item(y, x, 2))
        
        '''
        cv2.imshow('diffusion step 1', image)
        cv2.imwrite('diffusion_1.png', image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        '''

        boundary = n / 2
        targetImg = np.zeros((self.img.shape[0], self.img.shape[1], 3), np.uint8)
        for height in xrange(imgHeight):
            for width in xrange(imgWidth):
                filterB = []
                filterG = []
                filterR = []
                
                if height < boundary:
                    heightRangeStart = height
                    heightRangeEnd = n + height
                elif height > imgHeight - boundary - 1:
                    heightRangeStart = height - n
                    heightRangeEnd = height
                else:
                    heightRangeStart = height - boundary
                    heightRangeEnd = height + boundary + 1
                	
                if width < boundary:
                    widthRangeStart = width
                    widthRangeEnd = n + width
                elif width > imgWidth - boundary - 1:
                    widthRangeStart = width - n
                    widthRangeEnd = width
                else:
                    widthRangeStart = width - boundary
                    widthRangeEnd = width + boundary + 1
                    
                for x in xrange(heightRangeStart, heightRangeEnd):
                    for y in xrange(widthRangeStart, widthRangeEnd):
                        filterB.append(tmpImg.item(x, y, 0))
                        filterG.append(tmpImg.item(x, y, 1))
                        filterR.append(tmpImg.item(x, y, 2))

                medianB = np.median(filterB)
                medianG = np.median(filterG)
                medianR = np.median(filterR)
                targetImg.itemset((height, width, 0), medianB)
                targetImg.itemset((height, width, 1), medianG)
                targetImg.itemset((height, width, 2), medianR)
        '''
        cv2.imshow('diffusion step 2', targetImg)
        cv2.imwrite('diffusion_2.png', targetImg)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        '''
        return targetImg
