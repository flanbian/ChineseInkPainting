#!/usr/bin/python
# -*- coding: utf-8 -*-

import cv2
import numpy as np
import math

class Vec:
    def __init__(self, n):
        self.N = n
        self.p = [0 for y in range(n)]
        
    def make_uint(self):
        sum_p = 0.0
        for i in xrange(self.N):
            sum_p = sum_p + self.p[i] * self.p[i]
        sum_p = math.sqrt(sum_p)
        if sum_p > 0.0:
            for i in xrange(self.N):
                self.p[i] = self.p[i] / sum_p

    def normalization(self):
        sum_p = 0.0
        for i in xrange(self.N):
            sum_p = sum_p + self.p[i] * self.p[i]
        sum_p = math.sqrt(sum_p)
        return sum_p
        
    def zero(self):
        for i in xrange(self.N):
            self.p[i] = 0.0
        

class FDoG:
    def __init__(self):
        self.PI = 3.1415926

    def gauss(self, x, mean, sigma):
        return (math.exp((-(x - mean) * (x - mean)) / (2.0 * sigma * sigma) ) / math.sqrt(self.PI * 2.0 * sigma * sigma));

    def MakeGaussianVector(self, sigma):
        threshold = 0.001;
        
        i = 0
        while(True):
            i = i + 1
            if self.gauss(i, 0.0, sigma) < threshold:
                break
        GAU = Vec(i + 1)
        GAU.zero()
        
        GAU.p[0] = self.gauss(0.0, 0.0, sigma)
        for j in xrange(1, GAU.N):
            GAU.p[j] = self.gauss(j, 0.0, sigma)
            
        return GAU

    def GetDirectionalDoG(self, image, e, GAU1, GAU2, tau):
        vn = Vec(2)
        
        half_w1 = GAU1.N - 1
        half_w2 = GAU2.N - 1
        
        image_x = image.shape[0]
        image_y = image.shape[1]
        
        dog = [[0 for x in range(image_y)] for y in range(image_x)]
        
        for i in xrange(image_x):
            for j in xrange(image_y):
                sum1 = sum2 = 0.0
                w_sum1 = w_sum2 = 0.0
                weight1 = weight2 = 0.0
                
                vn.p[0] = -e.p[i][j].ty
                vn.p[1] = e.p[i][j].tx
                
                if vn.p[0] == 0.0 and vn.p[1] == 0.0:
                    sum1 = 255.0
                    sum2 = 255.0
                    dog[i][j] = sum1 - tau * sum2;
                    continue
                    
                d_x = i
                d_y = j
                
                for s in xrange(-half_w2, half_w2 + 1):
                    x = d_x + vn.p[0] * s
                    y = d_y + vn.p[1] * s
                    
                    if x > image_x - 1.0 or x < 0.0 or y > image_y - 1.0 or y < 0.0:
                        continue
                        
                    x1 = int(round(x))
                    if x1 < 0:
                        x1 = 0
                    if x1 > image_x - 1:
                        x1 = image_x - 1

                    y1 = int(round(y))
                    if y1 < 0:
                        y1 = 0
                    if y1 > image_y - 1:
                        y1 = image_y - 1
                        
                    val = image[x1][y1]
                    
                    dd = abs(s);
                    
                    weight1 = 0.0 if dd > half_w1 else GAU1.p[dd]
                    '''
                    if dd > half_w1:
				        weight1 = 0.0
				    else:
				        weight1 = GAU1.p[dd]
				    '''
                    sum1 = sum1 + val * weight1
                    w_sum1 = w_sum1 + weight1
				    
                    weight2 = GAU2.p[dd]
                    sum2 = sum2 + val * weight2
                    w_sum2 = w_sum2 + weight2
				    
                sum1 = sum1 / w_sum1
                sum2 = sum2 / w_sum2
				
                dog[i][j] = sum1 - tau * sum2
        return dog
        
    def GetFlowDoG(self, e, dog, GAU3):
        vt = Vec(2)
        
        image_x = len(dog)
        image_y = len(dog[0])
        
        tmp = [[0 for x in range(image_y)] for y in range(image_x)]
        
        half_l = GAU3.N - 1
        step_size = 1.0
        
        for i in xrange(image_x):
            for j in xrange(image_y):
                sum1 = 0.0
                w_sum1 = 0.0
                weight1 = 0.0
                
                val = dog[i][j]
                weight1 = GAU3.p[0]
                sum1 = val * weight1
                w_sum1 = w_sum1 + weight1
                
                d_x = i_x = i
                d_y = i_y = j
                
                for k in xrange(half_l):
                    vt.p[0] = e.p[i_x][i_y].tx
                    vt.p[1] = e.p[i_x][i_y].ty
                    if vt.p[0] == 0.0 and vt.p[1] == 0.0:
                        break
                    x = d_x
                    y = d_y
                    
                    if x > image_x - 1.0 or x < 0.0 or y > image_y - 1.0 or y < 0.0:
                        break
                    x1 = int(round(x))
                    if x1 < 0:
                        x1 = 0
                    if x1 > image_x - 1:
                        x1 = image_x - 1
                    
                    y1 = int(round(y))
                    if y1 < 0:
                        y1 = 0
                    if y1 > image_y - 1:
                        y1 = image_y - 1
                    
                    val = dog[x1][y1]
                    
                    weight1 = GAU3.p[k]
                    
                    sum1 = sum1 + val * weight1
                    w_sum1 = w_sum1 + weight1
                    
                    d_x = d_x + vt.p[0] * step_size; 
                    d_y = d_y + vt.p[1] * step_size; 
                    
                    i_x = int(round(d_x))
                    i_y = int(round(d_y))
                    if d_x < 0 or d_x > image_x - 1 or d_y < 0 or d_y > image_y - 1:
                        break
                d_x = i_x = i
                d_y = i_y = j
                
                for k in xrange(half_l):
                    vt.p[0] = -e.p[i_x][i_y].tx
                    vt.p[1] = -e.p[i_x][i_y].ty
                    if vt.p[0] == 0.0 and vt.p[1] == 0.0:
                        break
                    x = d_x
                    y = d_y
                    
                    if x > image_x - 1 or x < 0.0 or y > image_y - 1 or y < 0.0:
                        break;
                    x1 = int(round(x))
                    if x1 < 0:
                        x1 = 0
                    if x1 > image_x - 1:
                        x1 = image_x - 1
                        
                    y1 = int(round(y))
                    if y1 < 0:
                        y1 = 0
                    if y1 > image_y - 1:
                        y1 = image_y - 1
                    
                    val = dog[x1][y1]
                    weight1 = GAU3.p[k] 
                    
                    sum1 = sum1 + val * weight1
                    w_sum1 = w_sum1 + weight1
                    
                    d_x = d_x + vt.p[0] * step_size
                    d_y = d_y + vt.p[1] * step_size
                    
                    i_x = int(round(d_x))
                    i_y = int(round(d_y))
                    if d_x < 0 or d_x > image_x - 1 or d_y < 0 or d_y > image_y - 1:
                        break
                sum1 = sum1 / w_sum1; 

                tmp[i][j] = 1.0 if sum1 > 0 else 1.0 + np.tanh(sum1)
                '''
                if sum1 > 0:
                    tmp[i][j] = 1.0
                else:
                    tmp[i][j] = 1.0 + tanh(sum1)
                '''
        return tmp

    def getFDoG(self, image, e, sigma, sigma_3, tau):
        image_x = image.shape[0]
        image_y = image.shape[1]
        
        GAU1 = self.MakeGaussianVector(sigma)
        GAU2 = self.MakeGaussianVector(sigma * 1.6)
        half_w1 = GAU1.N - 1
        half_w2 = GAU2.N - 1
        
        GAU3 = self.MakeGaussianVector(sigma_3)
        half_l = GAU3.N - 1
        
        dog = self.GetDirectionalDoG(image, e, GAU1, GAU2, tau)
        tmp = self.GetFlowDoG(e, dog, GAU3)
        
        for i in xrange(image_x):
            for j in xrange(image_y):
                image[i][j] = int(round(tmp[i][j] * 255.0))
                
        return image
    	  
    def GrayThresholding(self, image, thres):
        image_x = image.shape[0]
        image_y = image.shape[1]
        
        for i in xrange(image_x):
            for j in xrange(image_y):
                val = image[i][j] / 255.0
                image[i][j] = int(round(val * 255.0)) if val < thres else 255
                
        return image