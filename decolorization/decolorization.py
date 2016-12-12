#!/usr/bin/python
# -*- coding: utf-8 -*-

import cv2
import numpy as np

class Decolorization:
    def __init__(self, image):
        self.img = image

    def decolorization(self, low, high):
        self.img = cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY)
        img = np.float64(self.img)
        
        shape = self.img.shape
        imgHeight = shape[0]
        imgWidth = shape[1]
        for x in xrange(imgHeight):
            for y in xrange(imgWidth):
                c = self.img.item(x, y)
                #print c
                if c < low:
                    img.itemset((x, y), 0)
                elif c > high:
                    img.itemset((x, y), 1)
                else:
                    img.itemset((x, y), ((c - low) / (high - low)))
        
        for x in xrange(imgHeight):
            for y in xrange(imgWidth):
                c = img.item(x, y)
                self.img.itemset((x, y), (c * 255))

        return self.img
