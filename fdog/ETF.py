#!/usr/bin/python
# -*- coding: utf-8 -*-

import cv2
import numpy as np
import math

class Vect:
    def __init__(self):
        self.tx = 1.0
        self.ty = 0.0
        self.mag = 1.0

class ETF:
    def __init__(self, Nr=3, Nc=2):
        self.Nr = Nr
        self.Nc = Nc
        self.max_grad = 1.0
        self.p = [[0 for x in range(Nc)] for y in range(Nr)]
        
        for x in xrange(Nr):
            for y in xrange(Nc):
                self.p[x][y] = Vect()

    def copy_ETF(self, etf):
        for i in xrange(self.Nr):
            for j in xrange(self.Nc):
				self.p[i][j].tx = etf.p[i][j].tx
				self.p[i][j].ty = etf.p[i][j].ty
				self.p[i][j].mag = etf.p[i][j].mag
        self.max_grad = etf.max_grad

    def make_uint(self, vx, vy):
        mag = math.sqrt(vx * vx + vy * vy)
        if mag != 0.0:
            vx = vx / mag
            vy = vy / mag
        return (vx, vy)
                
    def normalize(self):
        for i in xrange(self.Nr):
            for j in xrange(self.Nc):
                self.p[i][j].tx, self.p[i][j].ty = self.make_uint(self.p[i][j].tx, self.p[i][j].ty)
                '''
                mag = math.sqrt(self.p[i][j].tx * self.p[i][j].tx + self.p[i][j].ty * self.p[i][j].ty)
                if mag != 0.0:
                    self.p[i][j].tx = self.p[i][j].tx / mag
                    self.p[i][j].ty = self.p[i][j].ty / mag
                '''
                self.p[i][j].mag = self.p[i][j].mag / self.max_grad
                            

    def setFlow(self, image, MAX_VAL = 1020.0):
        self.max_grad = -1.0
        
        v = [0.0, 0.0]
        
        for i in xrange(1, self.Nr - 1):
            for j in xrange(1, self.Nc - 1):
                self.p[i][j].tx = (image[i+1][j-1] + 2.0*image[i+1][j] + image[i+1][j+1] - image[i-1][j-1] - 2.0*image[i-1][j] - image[i-1][j+1]) / MAX_VAL
                self.p[i][j].ty = (image[i-1][j+1] + 2.0*image[i][j+1] + image[i+1][j+1] - image[i-1][j-1] - 2.0*image[i][j-1] - image[i+1][j-1]) / MAX_VAL
                v[0] = self.p[i][j].tx
                v[1] = self.p[i][j].ty
                self.p[i][j].tx = -v[1]
                self.p[i][j].ty = v[0]
                
                self.p[i][j].mag = math.sqrt(self.p[i][j].tx * self.p[i][j].tx + self.p[i][j].ty * self.p[i][j].ty)
                
                if self.p[i][j].mag > self.max_grad:
                    self.max_grad = self.p[i][j].mag;
        
        for i in xrange(1, self.Nr - 2):
		    self.p[i][0].tx = self.p[i][1].tx
		    self.p[i][0].ty = self.p[i][1].ty
		    self.p[i][0].mag = self.p[i][1].mag;
		    self.p[i][self.Nc - 1].tx = self.p[i][self.Nc - 2].tx
		    self.p[i][self.Nc - 1].ty = self.p[i][self.Nc - 2].ty
		    self.p[i][self.Nc - 1].mag = self.p[i][self.Nc - 2].mag

        for j in xrange(1, self.Nc - 2):
            self.p[0][j].tx = self.p[1][j].tx
            self.p[0][j].ty = self.p[1][j].ty
            self.p[0][j].mag = self.p[1][j].mag
            self.p[self.Nr - 1][j].tx = self.p[self.Nr - 2][j].tx
            self.p[self.Nr - 1][j].ty = self.p[self.Nr - 2][j].ty
            self.p[self.Nr - 1][j].mag = self.p[self.Nr - 2][j].mag
            
        self.p[0][0].tx = ( self.p[0][1].tx + self.p[1][0].tx ) / 2;
        self.p[0][0].ty = ( self.p[0][1].ty + self.p[1][0].ty ) / 2;
        self.p[0][0].mag = ( self.p[0][1].mag + self.p[1][0].mag ) / 2;
        self.p[0][self.Nc-1].tx = ( self.p[0][self.Nc-2].tx + self.p[1][self.Nc-1].tx ) / 2;
        self.p[0][self.Nc-1].ty = ( self.p[0][self.Nc-2].ty + self.p[1][self.Nc-1].ty ) / 2;
        self.p[0][self.Nc-1].mag = ( self.p[0][self.Nc-2].mag + self.p[1][self.Nc-1].mag ) / 2;
        self.p[self.Nr-1][0].tx = ( self.p[self.Nr-1][1].tx + self.p[self.Nr-2][0].tx ) / 2;
        self.p[self.Nr-1][0].ty = ( self.p[self.Nr-1][1].ty + self.p[self.Nr-2][0].ty ) / 2;
        self.p[self.Nr-1][0].mag = ( self.p[self.Nr-1][1].mag + self.p[self.Nr-2][0].mag ) / 2;
        self.p[self.Nr - 1][self.Nc - 1].tx = ( self.p[self.Nr - 1][self.Nc - 2].tx + self.p[self.Nr - 2][self.Nc - 1].tx ) / 2;
        self.p[self.Nr - 1][self.Nc - 1].ty = ( self.p[self.Nr - 1][self.Nc - 2].ty + self.p[self.Nr - 2][self.Nc - 1].ty ) / 2;
        self.p[self.Nr - 1][self.Nc - 1].mag = ( self.p[self.Nr - 1][self.Nc - 2].mag + self.p[self.Nr - 2][self.Nc - 1].mag ) / 2;
        
        self.normalize()
    
    def smooth(self, half_w, M):
        MAX_GRADIENT = -1
        
        image_x = self.Nr
        image_y = self.Nc
        
        e2 = ETF(image_x, image_y)
        e2.copy_ETF(self)
        
        g = [0, 0]
        v = [0, 0]
        w = [0, 0]
        
        for k in xrange(M):
            for j in xrange(image_y):
                for i in xrange(image_x):
                    g[0] = g[1] = 0.0
                    v[0] = self.p[i][j].tx
                    v[1] = self.p[i][j].ty
                    for s in xrange(-half_w, half_w + 1):
                        x = i + s
                        y = j
                        if x > image_x - 1:
                            x = image_x - 1
                        elif x < 0:
                            x = 0
                        
                        if y > image_y - 1:
                            y = image_y - 1
                        elif y < 0:
                            y = 0
                            
                        mag_diff = self.p[x][y].mag - self.p[i][j].mag
                        
                        w[0] = self.p[x][y].tx
                        w[1] = self.p[x][y].ty
                        
                        factor = 1.0
                        angle = v[0] * w[0] + v[1] * w[1]
                        if angle < 0.0:
                            factor = -1.0
                            
                        weight = mag_diff + 1
                        g[0] = g[0] + weight * self.p[x][y].tx * factor
                        g[1] = g[1] + weight * self.p[x][y].ty * factor

                    g[0], g[1] = self.make_uint(g[0], g[1])
                    '''
                    mag = math.sqrt(g[0] * g[0] + g[1] * g[1])
                    if mag != 0.0:
                        g[0] = g[0] / mag
                        g[1] = g[1] / mag
                    '''
                    e2.p[i][j].tx = g[0]
                    e2.p[i][j].ty = g[1]
            self.copy_ETF(e2)
            
            for j in xrange(image_y):
                for i in xrange(image_x):
                    g[0] = g[1] = 0.0
                    v[0] = self.p[i][j].tx
                    v[1] = self.p[i][j].ty
                    for t in xrange(-half_w, half_w + 1):
                        x = i
                        y = j + t
                        if x > image_x - 1:
                            x = image_x - 1
                        elif x < 0:
                            x = 0
                        
                        if y > image_y - 1:
                            y = image_y - 1
                        elif y < 0:
                            y = 0
                            
                        mag_diff = self.p[x][y].mag - self.p[i][j].mag
                        
                        w[0] = self.p[x][y].tx
                        w[1] = self.p[x][y].ty
                        
                        factor = 1.0
                        angle = v[0] * w[0] + v[1] * w[1]
                        if angle < 0.0:
                            factor = -1.0
                            
                        weight = mag_diff + 1
                        g[0] = g[0] + weight * self.p[x][y].tx * factor
                        g[1] = g[1] + weight * self.p[x][y].ty * factor
                        
                    g[0], g[1] = self.make_uint(g[0], g[1])
                    '''
                    mag = math.sqrt(g[0] * g[0] + g[1] * g[1])
                    if mag != 0.0:
                        g[0] = g[0] / mag
                        g[1] = g[1] / mag
                    '''
                    e2.p[i][j].tx = g[0]
                    e2.p[i][j].ty = g[1]
            self.copy_ETF(e2)