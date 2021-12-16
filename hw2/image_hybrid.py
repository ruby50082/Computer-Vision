import cv2 
import numpy as np
import math
from numpy.fft import fft2, ifft2, fftshift, ifftshift
from matplotlib import pyplot as plt
from pathlib import Path

def make_filter(rows, cols, islowpass, sigma):
    o_x = int(rows/2) if rows%2 == 0 else int(rows/2) + 1 
    o_y = int(cols/2) if cols%2 == 0 else int(cols/2) + 1
    def gaussian(x,y):
        coef = math.exp(-1.0 * ((x - o_x)**2+(y - o_y)**2)/(2 * sigma ** 2))
        return coef if islowpass else 1 - coef
    def ideal(x,y):
        near_center = True if math.sqrt((x - o_x)**2+(y - o_y)**2) <= sigma else False 
        return 1 if (islowpass and near_center) or (not islowpass and not near_center) else 0
    func = ideal
    return np.array([[func(i,j)for j in range(cols)] for i in range(rows)] ) 

def transform(img, islowpass, sigma):
    DFT = fftshift(fft2(img))
    mask = make_filter(img.shape[0], img.shape[1], islowpass, sigma)
    IDFT = ifft2(ifftshift(DFT * mask))
    filterd_img = np.real(IDFT)
    return filterd_img


if __name__ == '__main__':
    H_sig = 25
    L_sig = 15
    imgs = list(Path("hw2_data", "task1,2_hybrid_pyramid").glob('*'))
    sorted_imgs = sorted(imgs, key = lambda img: img.stem.split()[0])
    num = 0
    for img1, img2 in zip(sorted_imgs[1:-1:2], sorted_imgs[2::2]):
        print(num)
        high = cv2.imread(str(img1), cv2.IMREAD_GRAYSCALE)
        low = cv2.imread(str(img2), cv2.IMREAD_GRAYSCALE)

        highpass = transform(img = high, islowpass = False, sigma = H_sig)
        lowpass = transform(img = low, islowpass = True, sigma = L_sig)  

        fig = plt.figure(figsize=(14,5))
        fig.add_subplot(131,title = f'Low')
        plt.imshow(lowpass, cmap='gray')
        fig.add_subplot(132,title = f'High')
        plt.imshow(highpass, cmap='gray')

        if num == 6:
            if highpass.shape[0] > lowpass.shape[0]:
                highpass = highpass[:-1, :-1]
            else:
                lowpass = lowpass[:-1,:-1]
        res = highpass + lowpass
        fig.add_subplot(133,title = f'Hybrid')
        plt.imshow(res, cmap='gray')

        dir_path = f'hybrid_result/{num}'
        if not Path(dir_path).exists():
            Path(dir_path).mkdir()
        plt.savefig(dir_path + "/rec_i_exp_{n}_{sig_l}_{sig_h}.jpg".format(n=num, sig_l = L_sig, sig_h = H_sig))
        cv2.imwrite(dir_path + "/rec_i_{n}_{h_l}_{sig}.jpg".format(n=num, h_l = 'H', sig = H_sig), highpass)
        cv2.imwrite(dir_path + "/rec_i_{n}_{h_l}_{sig}.jpg".format(n=num, h_l = 'L', sig = L_sig), lowpass)
        cv2.imwrite(dir_path + "/rec_i_res_{n}_{sig_l}_{sig_h}.jpg".format(n=num, sig_h = H_sig, sig_l = L_sig), res)
        num += 1
    

    
    high = cv2.imread("own_data/IMG_0001.JPG",cv2.IMREAD_GRAYSCALE)
    low = cv2.imread("own_data/IMG_0002.JPG", cv2.IMREAD_GRAYSCALE)
    fig = plt.figure(figsize = (14,5))
    highpass = transform(high, False, H_sig)
    lowpass = transform(low, True, L_sig)  
    fig = plt.figure(figsize=(14,5))
    fig.add_subplot(131, title = 'Low')
    plt.imshow(lowpass, cmap='gray')
    fig.add_subplot(132, title = 'High')
    plt.imshow(highpass, cmap='gray')
    res = highpass + lowpass
    fig.add_subplot(133, title = 'Hybrid')
    plt.imshow(res, cmap='gray')
    plt.savefig("hybrid_result/i_exp_{n}_{sig_l}_{sig_h}.jpg".format(n='patrick', sig_l = L_sig, sig_h = H_sig))
    cv2.imwrite("hybrid_result/i_{n}_{h_l}_{sig}.jpg".format(n='patrick', h_l = 'H', sig = H_sig), highpass)
    cv2.imwrite("hybrid_result/i_{n}_{h_l}_{sig}.jpg".format(n='desert', h_l = 'L', sig = L_sig), lowpass)
    cv2.imwrite("hybrid_result/i_res_{n}_{sig_l}_{sig_h}.jpg".format(n='patrick', sig_h = H_sig, sig_l = L_sig), res)

    high = cv2.imread("hw2_data/task1,2_hybrid_pyramid/0_Afghan_girl_before.jpg",cv2.IMREAD_GRAYSCALE)
    low = cv2.imread("hw2_data/task1,2_hybrid_pyramid/0_Afghan_girl_before.jpg", cv2.IMREAD_GRAYSCALE)
    for sigma in [10, 20, 30, 40, 50]:
        highpass = transform(high, False, sigma)
        fig.add_subplot(1,5,int(sigma/10), title = f'cutoff_F = {sigma}')
        plt.imshow(highpass, cmap='gray')
    plt.savefig("sigma_comp_h.jpg")

    fig = plt.figure(figsize = (14,5))
    for sigma in [10, 20, 30, 40, 50]:
        lowpass = transform(low, True, sigma)
        fig.add_subplot(1,5,int(sigma/10), title = f'cutoff_F = {sigma}')
        plt.imshow(lowpass, cmap='gray')
    plt.savefig("sigma_comp_l.jpg")
    

    

