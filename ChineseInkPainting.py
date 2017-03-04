import cv2
import numpy as np

from SaliencyMap.pySaliencyMap import pySaliencyMap
from saliency_map.saliency_map import SaliencyMap
from ink_diffusion.diffusion import Diffusion
from dog import Dog
from decolorization import Decolorization

import os
from os import listdir
from os.path import isfile, join

#from matplotlib import pyplot as plt
#from saliency_map.utils import OpencvIo
#from skimage import io, color

SOURCE_PATH = "SourceImg/"
RESULT_PATH = "Result/"

sourceFiles = [f for f in listdir(SOURCE_PATH) 
        if isfile(join(SOURCE_PATH, f)) and f.endswith(".jpg") and not f.startswith("xuan")
]

print sourceFiles
for sourceFile in sourceFiles:

    img = cv2.imread(SOURCE_PATH + sourceFile)

    # SaliencyMap start
    print "SaliencyMap start"
    img_width  = img.shape[1]
    img_height = img.shape[0]
    sm = pySaliencyMap(img_width, img_height)
    saliencyMapImg = sm.SMGetSM(img)
    saliencyMapImg = sm.SMNormalization(saliencyMapImg)
    saliencyMapImg = np.uint8(cv2.cvtColor(saliencyMapImg, cv2.COLOR_GRAY2RGB))
    '''
    oi = OpencvIo()
    sm = SaliencyMap(img)
    saliencyMapImg = oi.imget_array([sm.map])
    saliencyMapImg = cv2.cvtColor(saliencyMapImg, cv2.COLOR_GRAY2RGB)
    '''
    print "SaliencyMap finish"
    # SaliencyMap end

    # AbstractionImg source = SaliencyMap + sourceImg
    saliencyMapAndSourceImg = cv2.addWeighted(saliencyMapImg, 0.2, img, 0.8, 0)
    '''
    cv2.imshow('saliencyMapAndSourceImg', saliencyMapAndSourceImg)
    cv2.waitKey()
    cv2.destroyAllWindows()
    '''
    # XDoG start
    print "XDoG start"
    dog = Dog(img)
    xdogImg = cv2.cvtColor(np.uint8(dog.xdogGrayTransform(dog.xdog())), cv2.COLOR_GRAY2RGB)
    '''
    cv2.imshow('xdogImg', xdogImg)
    cv2.waitKey()
    cv2.destroyAllWindows()
    '''
    print "XDoG finish"
    # XDoG finish

    # Abstraction start
    print "Abstraction start"
    saliencyMapAndSourceImg= cv2.addWeighted(saliencyMapAndSourceImg, 0.9, xdogImg, 0.1, 0)

    abstractImg = cv2.bilateralFilter(saliencyMapAndSourceImg, -1, 2, 30)
    '''
    cv2.imshow('abstractImg', abstractImg)
    cv2.waitKey()
    cv2.destroyAllWindows()
    '''
    print "Abstraction finish"
    # Abstraction end

    # Diffusion start
    print "Diffusion start"
    diffusion = Diffusion(abstractImg)
    diffusionImg = diffusion.diffusion(5, 5)
    '''
    cv2.imshow('diffusionImg', diffusionImg)
    cv2.waitKey()
    cv2.destroyAllWindows()
    '''
    print "Diffusion finish"
    # Diffusion finish

    # Decolorization start
    print "Decolorization start"
    decolorization = Decolorization(diffusionImg)
    decolorizationImg = decolorization.decolorization(20.0, 200.0)
    decolorizationImg = cv2.cvtColor(decolorizationImg, cv2.COLOR_GRAY2RGB)
    '''
    cv2.imshow('decolorizationImg', decolorizationImg)
    cv2.waitKey()
    cv2.destroyAllWindows()
    '''
    print "Decolorization finish"
    # Decolorization finish

    # Combine detail start
    print "Combine detail start"
    dog = Dog(img)
    secondXdogImg = cv2.cvtColor(np.uint8(dog.xdogGrayTransform(dog.xdog(1.0, 2.0))), cv2.COLOR_GRAY2RGB)
    xdogImgBlur = cv2.blur(secondXdogImg, (2, 2))
    edgeCombinationImg = cv2.addWeighted(xdogImgBlur, 0.1, decolorizationImg, 0.9, 0)
    print "Combine detail finish"
    # Combine detail finish

    # Paper Texture start
    print "Paper Texture start"
    height, width = img.shape[:2]
    textureImg = cv2.imread("SourceImg/xuan_paper.jpg")
    textureImg = cv2.resize(textureImg, (width, height), interpolation = cv2.INTER_CUBIC)
    textureImg = cv2.addWeighted(textureImg, 0.1, edgeCombinationImg, 0.9, 0)
    print "Paper Texture finish"
    # Paper Texture finish

    # Second decolorization start
    print "Second decolorization start"
    decolorization = Decolorization(textureImg)
    SecondDecolorationImg = decolorization.decolorization(20.0, 200.0)
    SecondDecolorationImg = cv2.cvtColor(SecondDecolorationImg, cv2.COLOR_GRAY2RGB)
    print "Second decolorization finish"
    # Second decolorization end

    resultPath = RESULT_PATH + sourceFile[:-4] + "/"
    if not os.path.exists(resultPath):
        os.makedirs(resultPath)

    cv2.imwrite(resultPath + "saliencyMap.png", saliencyMapImg)
    cv2.imwrite(resultPath + "saliencyMapAndSourceImg.png", saliencyMapAndSourceImg)
    cv2.imwrite(resultPath + "abstractImg.png", abstractImg)
    cv2.imwrite(resultPath + "diffusionImg.png", diffusionImg)
    cv2.imwrite(resultPath + "xdogImg.png", xdogImg)
    cv2.imwrite(resultPath + "xdogImgBlur.png", xdogImgBlur)
    cv2.imwrite(resultPath + "edgeCombinationImg.png", edgeCombinationImg)
    cv2.imwrite(resultPath + "decolorizationImg.png", decolorizationImg)
    cv2.imwrite(resultPath + "textureImg.png", textureImg)
    cv2.imwrite(resultPath + "SecondDecolorationImg.png", SecondDecolorationImg)
