#!/usr/bin/python
# -*- coding: utf-8 -*-

import cv2
import numpy as np

class Dog:
    def __init__(self, image):
        self.img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    ## Sharp image from scaled DoG signal.
    #  @param  sigma      sigma for small Gaussian filter.
    #  @param  k_sigma    large/small sigma (Gaussian filter).
    #  @param  p          scale parameter for DoG signal to make sharp.
    def sharpImage(self, sigma, k_sigma, p):
        sigma_large = sigma * k_sigma
        G_small = cv2.GaussianBlur(self.img, (0, 0), sigma)
        G_large = cv2.GaussianBlur(self.img, (0, 0), sigma_large)
        
        S = (1 + p) * G_small - p * G_large
        return S

    ## Soft threshold function to make ink rendering style.
    #  @param  epsilon    threshold value between dark and bright.
    #  @param  phi        soft thresholding parameter.
    def softThreshold(self, SI, epsilon, phi):
        T = np.zeros(SI.shape)
        SI_bright = SI >= epsilon
        SI_dark = SI < epsilon
        T[SI_bright] = 1.0
        T[SI_dark] = 1.0 + np.tanh( phi * (SI[SI_dark] - epsilon))
        return T

    ## XDoG filter.
    #  @param  img        input gray image.
    #  @param  sigma      sigma for sharpImage.
    #  @param  k_sigma    large/small sigma for sharpImage.
    #  @param  p          scale parameter for sharpImage.
    #  @param  epsilon    threshold value for softThreshold.
    #  @param  phi        soft thresholding parameter for softThreshold.
    def xdog(self, sigma=1.0, k_sigma=31.0, p=21.7, epsilon=1.0, phi=1.0):
    	  S = self.sharpImage(sigma, k_sigma, p)
    	  SI = np.multiply(self.img, S)
    	  T = self.softThreshold(SI, epsilon, phi)
    	  
    	  return T

    def xdogGrayTransform(self, img):
    	  shape = img.shape
    	  height = shape[0]
    	  width = shape[1]
    	  for x in range(height):
    	  	  for y in range(width):
    	  	  	  val = img.item(x, y)
    	  	  	  img.itemset((x, y), val * 200)
    	  	  	  
    	  return img
    	  
    def xdogColorTransform(self, img):
    	  shape = img.shape
    	  height = shape[0]
    	  width = shape[1]
    	  for x in range(height):
    	  	  for y in range(width):
    	  	  	  B = img.item(x, y, 0)
    	  	  	  G = img.item(x, y, 1)
    	  	  	  R = img.item(x, y, 2)
    	  	  	  img.itemset((x, y, 0), B * 255)
    	  	  	  img.itemset((x, y, 1), G * 255)
    	  	  	  img.itemset((x, y, 2), R * 255)

    	  return img