import cv2
import numpy as np

from SaliencyMap.pySaliencyMap import pySaliencyMap
from saliency_map.saliency_map import SaliencyMap
from ink_diffusion.diffusion import Diffusion
from dog import Dog
from fdog import *
from decolorization import Decolorization

import os
from os import listdir
from os.path import isfile, join

#from matplotlib import pyplot as plt
#from saliency_map.utils import OpencvIo
#from skimage import io, color

SOURCE_PATH = "SourceImg/dragon/"
RESULT_PATH = "Result/dragon/"

sourceFiles = [f for f in listdir(SOURCE_PATH) 
        if isfile(join(SOURCE_PATH, f)) and f.endswith(".jpg") and not f.startswith("xuan")
]

print sourceFiles

def saveImg(fileName, Image):
    global RESULT_PATH
    global sourceFile
    resultPath = RESULT_PATH + sourceFile[:-4] + "/"
    if not os.path.exists(resultPath):
        os.makedirs(resultPath)
        
    cv2.imwrite(resultPath + fileName, Image)

def getSaliencyMapImg(image):
    print "SaliencyMap start"
    img_width  = image.shape[1]
    img_height = image.shape[0]
    sm = pySaliencyMap(img_width, img_height)
    saliencyMapImg = sm.SMGetSM(image)
    saliencyMapImg = sm.SMNormalization(saliencyMapImg)
    saliencyMapImg = np.uint8(cv2.cvtColor(saliencyMapImg, cv2.COLOR_GRAY2RGB))
    saveImg("saliencyMapImg.png", saliencyMapImg)
    print "SaliencyMap finish"
    return saliencyMapImg
    
def getAbstractionImg(image):
    print "Abstraction start"
    abstractImg = cv2.bilateralFilter(image, -1, 2, 30)
    saveImg("abstractImg.png", abstractImg)
    print "Abstraction finish"
    return abstractImg

def getDiffusionImg(image):
    print "Diffusion start"
    diffusion = Diffusion(image)
    diffusionImg = diffusion.diffusion(5, 5)
    saveImg("diffusionImg.png", diffusionImg)
    print "Diffusion finish"
    return diffusionImg
    
def getXDoGImg(image, sigma=1.0, k_sigma=3.0):
    print "XDoG start"
    dog = Dog(image)
    XDoGImg = cv2.cvtColor(np.uint8(dog.xdogGrayTransform(dog.xdog(sigma, k_sigma))), cv2.COLOR_GRAY2RGB)
    saveImg("XDoGImg.png", XDoGImg)
    print "XDoG finish"
    return XDoGImg
    
def getDecolorizationImg(image, dark=20.0, light=200.0):
    print "Decolorization start"
    decolorization = Decolorization(image)
    decolorizationImg = decolorization.decolorization(dark, light)
    decolorizationImg = cv2.cvtColor(decolorizationImg, cv2.COLOR_GRAY2RGB)
    saveImg("decolorizationImg.png", decolorizationImg)
    print "Decolorization finish"
    return decolorizationImg
    
def getFDoGImg(image):
    print "FDoG start"
    img_gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    e = ETF(img_gray.shape[0], img_gray.shape[1])
    e.setFlow(img_gray)
    e.smooth(4, 2)
    
    fdog = FDoG()
    FDoGImg = fdog.getFDoG(img_gray, e, 1.0, 3.0, 0.97)
    FDoGImg = fdog.GrayThresholding(FDoGImg, 0.7)
    saveImg("FDoGImg.png", FDoGImg)
    print "FDoG finish"
    return FDoGImg
    
def getTextureImg(image):
    print "Paper Texture start"
    height, width = image.shape[:2]
    textureImg = cv2.imread("SourceImg/xuan_paper.jpg")
    textureImg = cv2.resize(textureImg, (width, height), interpolation = cv2.INTER_CUBIC)
    print "Paper Texture finish"
    return textureImg

def getMedianBlur(image, sigma=3):
    print "Median Blur start"
    medianBlurImg = cv2.medianBlur(image, sigma)
    saveImg("medianBlurImg.png", medianBlurImg)
    print "Median Blur finish"
    return medianBlurImg
    
def getMeansDenoising(image, sigma=5):
    print "Means Denoising start"
    meansDenoiseImg = cv2.fastNlMeansDenoisingColored(image, None, sigma, sigma, 7, 21)
    saveImg("meansDenoiseImg.png", meansDenoiseImg)
    print "Means Denoising finish"
    return meansDenoiseImg
    
def combineTwoImageWithBlack(target_image, combine_image, ratio, image_name):
    for x in xrange(target_image.shape[0]):
        for y in xrange(target_image.shape[1]):
            if combine_image[x][y] == 0:
                target_image[x][y] = target_image[x][y] * ratio + combine_image[x][y] * (1 - ratio)
    
    saveImg(image_name, target_image)
    return target_image
    

for sourceFile in sourceFiles:

    resultPath = RESULT_PATH + sourceFile[:-4] + "/"

    img = cv2.imread(SOURCE_PATH + sourceFile)
    #image = getXDoGImg(img, 1.0, 3.0)
    FDoGImage = getFDoGImg(img)
    textureImage = getTextureImg(img)
    
    image = getSaliencyMapImg(img)
    image = cv2.addWeighted(image, 0.2, img, 0.8, 0)
    image = getAbstractionImg(image)
    image = getDiffusionImg(image)
    image = getDecolorizationImg(image)
    image = combineTwoImageWithBlack(image, FDoGImage, 0.8, "detail.png")
    #image = cv2.addWeighted(textureImage, 0.2, image, 0.8, 0)
    #saveImg("textureImage.png", image)
    #image = getMedianBlur(image, 3)
    #image = getMedianBlur(image, 3)
    #image = getMeansDenoising(image, sigma)
