import cv2
import numpy as np

from SaliencyMap.pySaliencyMap import pySaliencyMap
from saliency_map.saliency_map import SaliencyMap
from ink_diffusion.diffusion import Diffusion
from dog import Dog
from decolorization import Decolorization


#from matplotlib import pyplot as plt
#from saliency_map.utils import OpencvIo
#from skimage import io, color

#oi = OpencvIo()

img = cv2.imread("SourceImg/6.jpg")
# SaliencyMap start
print "SaliencyMap start"
'''
sm = SaliencyMap(img)
saliencyMapImg = oi.imget_array([sm.map])
saliencyMapImg = cv2.cvtColor(saliencyMapImg, cv2.COLOR_GRAY2RGB)
'''
img_width  = img.shape[1]
img_height = img.shape[0]
sm = pySaliencyMap(img_width, img_height)
saliencyMapImg = sm.SMGetSM(img)
saliencyMapImg = sm.SMNormalization(saliencyMapImg)
saliencyMapImg = np.uint8(cv2.cvtColor(saliencyMapImg, cv2.COLOR_GRAY2RGB))

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

abstractImg = cv2.bilateralFilter(saliencyMapAndSourceImg, -1, 2, 20)
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
xdogImg = cv2.blur(xdogImg, (2, 2))
EdgeCombinationImg = cv2.addWeighted(xdogImg, 0.2, decolorizationImg, 0.8, 0)

# Paper Texture start
print "Paper Texture start"
height, width = img.shape[:2]
textureImg = cv2.imread("SourceImg/xuan_paper.jpg")
textureImg = cv2.resize(textureImg, (width, height), interpolation = cv2.INTER_CUBIC)
print "Paper Texture finish"
# Paper Texture finish

textureImg = cv2.addWeighted(textureImg, 0.2, EdgeCombinationImg, 0.8, 0)

cv2.imwrite("Result/saliencyMap.png", saliencyMapImg)
cv2.imwrite("Result/saliencyMapAndSourceImg.png", saliencyMapAndSourceImg)
cv2.imwrite("Result/abstractImg.png", abstractImg)
cv2.imwrite("Result/diffusionImg.png", diffusionImg)
cv2.imwrite("Result/xdogImg.png", xdogImg)
cv2.imwrite("Result/EdgeCombinationImg.png", EdgeCombinationImg)
cv2.imwrite("Result/decolorizationImg.png", decolorizationImg)
cv2.imwrite("Result/textureImg.png", textureImg)