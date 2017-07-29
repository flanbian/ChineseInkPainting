

# -*- coding: utf-8 -*-
## @package guided_filter.results.smooth_noise
#
#  Simple guided filter test.
#  @author      tody
#  @date        2015/08/26

import os
import numpy as np
import matplotlib.pyplot as plt

from guided_filter.datasets.google_image import dataFile
from guided_filter.results.results import resultFile
from guided_filter.io_util.image import loadRGB
from guided_filter.io_util.image import saveImage
from guided_filter.cv.image import to32F
from guided_filter.cv.image import to8U

from guided_filter.core.filters import FastGuidedFilter, GuidedFilter

import cv2

def runSmoothNoiseResult(image_file):
    '''
    image_name = os.path.basename(image_file)
    image_name = os.path.splitext(image_name)[0]
    
    C_8U = loadRGB(image_file)
    '''
    C_8U = image_file
    C_32F = to32F(C_8U)

    aspect = C_32F.shape[0] / float(C_32F.shape[1])

    fig_width = 10
    fig_height = int(2 * fig_width * aspect / 3) + 2
    fig = plt.figure(figsize=(fig_width, fig_height))
    fig.subplots_adjust(left=0.05, bottom=0.05, right=0.95, top=0.82, wspace=0.02, hspace=0.3)
    h, w = C_32F.shape[:2]
    image_size_str = "Image size: %s x %s" %(w, h)
    fig.suptitle("Filtering noise image\n%s" % image_size_str)

    plt.subplot(231)
    plt.title("Original")
    plt.imshow(C_32F)
    plt.axis('off')

    h, w, cs = C_32F.shape

    sigmas = [5, 10, 20]
    sigma = 20

    guided_filter = FastGuidedFilter(C_32F, radius=sigma, epsilon=0.02)
    C_smooth = guided_filter.filter(C_32F)
    C_smooth = np.clip(C_smooth, 0.0, 1.0)
    img_8U = to8U(C_smooth)
    return img_8U
    #saveImage(resultFile(image_name + str(sigma)), C_smooth)

def runSmoothNoiseResults(data_names, data_ids):
    for data_name in data_names:
        print "Smooth noise: %s" % data_name
        for data_id in data_ids:
            print "Data ID: %s" % data_id
            image_file = dataFile(data_name, data_id)
            runSmoothNoiseResult(image_file)


if __name__ == '__main__':
    data_names = ["flower"]
    data_ids = range(3)

    runSmoothNoiseResults(data_names, data_ids)
