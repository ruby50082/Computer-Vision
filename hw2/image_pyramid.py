import cv2
import math
import numpy as np

from pathlib import Path
from numpy.fft import fft2, ifft2, fftshift, ifftshift
from matplotlib import pyplot as plt
from image_hybrid import transform, make_filter

def upsample(img, factor):
    rows, cols = img.shape
    resize_img = np.zeros((rows*factor, cols*factor))
    for i in range(resize_img.shape[0]-1):
        for j in range(resize_img.shape[1]-1):
            org_x = round(i * (1/factor))
            org_y = round(j * (1/factor))
            resize_img[i,j] = img[org_x, org_y] 
    return resize_img


def make_gaussianPyr(img,factor):
    pyramid = []
    pyramid.append(img)
    spectrums = []
    DFT = fftshift(fft2(img))
    spectrums.append(20*np.log(abs(DFT)))
    for i in range(4):
        lowpass = transform(pyramid[-1], islowpass = True, sigma = 10)
        sub = lowpass[::factor,::factor].copy()
        pyramid.append(sub)
        
        DFT = fftshift(fft2(sub))
        spectrums.append(20 * np.log(np.abs(DFT)))

    return pyramid, spectrums

def make_laplacePyr(g_Pyr, factor):
    pyramid = []
    pyramid.append(g_Pyr[-1])
    spectrums = []
    DFT = fftshift(fft2(g_Pyr[-1]))
    spectrums.append(20*np.log(np.abs(DFT)))

    for Gi, Gi_1 in zip(g_Pyr[-2::-1], g_Pyr[-1:0:-1]):
        Gi_1_up = upsample(Gi_1,factor)
        if Gi_1_up.shape[0] > Gi.shape[0]:
            Gi_1_up = Gi_1_up[:-1]
        if Gi_1_up.shape[1] > Gi.shape[1]:
            Gi_1_up =  Gi_1_up[:,:-1]
        up = Gi - transform(Gi_1_up, islowpass=True, sigma = 10)
        pyramid.append(up)

        DFT = fftshift(fft2(up))
        spectrums.append(20*np.log(np.abs(DFT)))


    pyramid = pyramid[::-1]
    spectrums = spectrums[::-1]
    return pyramid, spectrums
    
if __name__ == '__main__':
    factor = 2 
    img = cv2.imread("own_data/IMG_4402.JPG", cv2.IMREAD_GRAYSCALE)
    g_pyramid, g_spectrums = make_gaussianPyr(img,factor)
    l_pyramid, l_spectrums = make_laplacePyr(g_pyramid,factor)

    dir_path = f'pyramid_result'
    if not Path(dir_path).exists():
        Path(dir_path).mkdir()

    fig1 = plt.figure(1,figsize = (15,8))
    for i in range(1, 6):
        fig1.add_subplot(2,5,i)
        plt.imshow(g_pyramid[i-1], cmap='gray')
    for i in range(6, 11):
        fig1.add_subplot(2,5,i)
        plt.imshow(g_spectrums[i-6])
    plt.savefig(dir_path + f'/g_pyr_{factor}')

    fig2 = plt.figure(2,figsize = (15,8))
    for i in range(1, 6):
        fig2.add_subplot(2,5,i)
        plt.imshow(l_pyramid[i-1], cmap='gray')
    for i in range(6, 11):
        fig2.add_subplot(2,5,i)
        plt.imshow(l_spectrums[i-6])
    plt.savefig(dir_path + f'/l_pyr_{factor}')




