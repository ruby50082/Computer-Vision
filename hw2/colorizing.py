import os
import cv2
import glob
import time
import numpy as np
from PIL import Image
from numpy.fft import fft2, ifft2

def preprocess(img):
    img = crop_img(img)
    blue, green, red = split_img(img)
    return blue, green, red 

def crop_img(img, top=0.03, down=0.03, left=0.06, right=0.06):
    h, w = img.shape
    img = img[int(h * top) : int(h - h * down), int(w * left) : int(w - w * right)]
    return img

def split_img(img):
    h, w = img.shape
    blue = img[0 : int(h/3), :]
    green = img[int(h/3): int(h/3) * 2, :]
    red = img[int(h/3) * 2: int(h/3) * 3, :]

    return blue, green, red

def alignment(img1, img2, offset, zncc=1):
    if zncc == 1:
        shift_pos = align_ZNCC(img1, img2, offset)
    else:
        shift_pos = align_SSD(img1, img2, offset)

    return shift_pos

### alignment w/ ZNCC ###
def align_ZNCC(img1, img2, offset):
    max_response = -1
    for i in range(-offset, offset):
        for j in range(-offset, offset):
            zncc_response = ZNCC(img1, np.roll(img2, [i, j], axis=(0, 1)))
            if zncc_response > max_response:
                max_response = zncc_response
                shift_pos = [i, j]

    return shift_pos

def ZNCC(img1, img2):
    img1 = img1 - img1.mean(axis=0)
    img2 = img2 - img2.mean(axis=0)
    img1_norm = img1 / np.linalg.norm(img1)
    img2_norm = img2 / np.linalg.norm(img2)

    return np.sum(img1_norm * img2_norm)

### alignment w/ SSD ###
def align_SSD(img1, img2, offset):

    max_response = float("inf")
    for i in range(-offset, offset):
        for j in range(-offset, offset):
            ncc_response = SSD(img1, np.roll(img2, [i, j], axis=(0, 1)))
            if ncc_response < max_response:
                max_response = ncc_response
                shift_pos = [i, j]

    return shift_pos
    
def SSD(img_1, img_2):
    ssd = np.sum((img_1 - img_2) ** 2)
    return ssd

def sobel(img):
    sobel_x = np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]])
    sobel_y = sobel_x.T

    img_x = conv(img, sobel_x)
    img_y = conv(img, sobel_y)
    img_x = np.uint8(np.absolute(img_x))
    img_y = np.uint8(np.absolute(img_y))
    filter_img = cv2.bitwise_or(img_x, img_y)

    return filter_img

def conv(A, B):
    return np.real(ifft2(fft2(A)*fft2(B, s=A.shape)))

def gaussian_kernel(sigma, truncate=4.0):

    sigma = float(sigma)
    radius = int(truncate * sigma + 0.5)

    x, y = np.mgrid[-radius:radius+1, -radius:radius+1]
    sigma = sigma**2

    k = 2*np.exp(-0.5 * (x**2 + y**2) / sigma)
    k = k / np.sum(k)

    return k

def init_ratio(w):
    if w < 1000:
        x = 5
    elif w < 2000:
        x = 10
    else:
        x = 20
    return int(x)


path = 'hw2_data/task3_colorizing/'
types = ('*.jpg', '*.tif') 
for _type in types:
    for _path in glob.glob(path + _type):
        start_time = time.time()

        img = Image.open(_path)
        img = np.asarray(img)
        height = int(img.shape[0] / 3)


        ### preprocess - crop and split ###
        blue, green, red = preprocess(img)

        if _type == '*.tif':
            green = green / 256
            red = red / 256
            blue = blue / 256

        ### filter ###
        filter_b, filter_g, filter_r = sobel(blue), sobel(green), sobel(red)

        ### pyramid ###
        crop_height = blue.shape[0]
        crop_width = blue.shape[1]
        ratio = init_ratio(crop_width)

        ratio_between_layer = 2
        offset = int((height // ratio) // 5)

        total_offset_g2b = np.zeros([2])
        total_offset_r2b = np.zeros([2])
        
        filter_b_ = conv(filter_b, gaussian_kernel(1))
        filter_g_ = conv(filter_g, gaussian_kernel(1))
        filter_r_ = conv(filter_r, gaussian_kernel(1))

        while(ratio >= 1):   ### small to large ###
            ### down sampling ###
            down_b = filter_b_[::ratio, ::ratio]
            down_g = filter_g_[::ratio, ::ratio]
            down_r = filter_r_[::ratio, ::ratio]

            ### align ###
            offset_g2b = alignment(down_b, down_g, offset, zncc=0)
            offset_r2b = alignment(down_b, down_r, offset, zncc=0)

            ### rolling ###
            offset_g2b = [element * ratio for element in offset_g2b]
            offset_r2b = [element * ratio for element in offset_r2b]
            
            filter_g_ = np.roll(filter_g_, offset_g2b, axis=(0,1))
            filter_r_ = np.roll(filter_r_, offset_r2b, axis=(0,1))

            total_offset_g2b = [a + b for a, b in zip(total_offset_g2b, offset_g2b)]
            total_offset_r2b = [a + b for a, b in zip(total_offset_r2b, offset_r2b)]

            ratio = int(ratio / ratio_between_layer)
            offset = int(offset / ratio_between_layer)

        ### final rolling ###
        total_offset_g2b = [int(a) for a in total_offset_g2b]
        total_offset_r2b = [int(a) for a in total_offset_r2b]

        shift_green = np.roll(green, total_offset_g2b, axis=(0,1))
        shift_red = np.roll(red, total_offset_r2b, axis=(0,1))

        ### form RGB image ###
        rgb_img = (np.dstack((shift_red, shift_green, blue))).astype(np.uint8)

        top_left_x = max(total_offset_g2b[0], total_offset_r2b[0], 0)
        top_left_y = max(total_offset_g2b[1], total_offset_r2b[1], 0)
        
        bottom_right_x = min(total_offset_g2b[0] + crop_height, total_offset_r2b[0] + crop_height, crop_height)
        bottom_right_y = min(total_offset_g2b[1] + crop_width, total_offset_r2b[1] + crop_width, crop_width)
        cropped_rgb_img = rgb_img[top_left_x:bottom_right_x, top_left_y:bottom_right_y, :]
        rgb_img = Image.fromarray(cropped_rgb_img)

        output_path = 'hw2_data/task3_demo'
        if not os.path.exists(output_path):
            os.makedirs(output_path)

        img_name = _path.split('/')
        print(img_name[-1], total_offset_g2b, total_offset_r2b)
        # print("--- %s seconds ---" % (time.time() - start_time))
        rgb_img.save(output_path + '/output_' + img_name[-1])


