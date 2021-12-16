import cv2
import sys
import numpy as np

from draw import *
from match import *
from RANSAC import *
from triangular import *  
    
if __name__ == '__main__':
    case = sys.argv[1] 
    if case == "1":
        InputFile1 = "Mesona1.JPG"
        InputFile2 = "Mesona2.JPG"
    elif case == "2":
        InputFile1 = "Statue1.bmp"
        InputFile2 = "Statue2.bmp"
    elif case == "3":
        InputFile1 = "ryan3.JPG"
        InputFile2 = "ryan4.JPG"

    filename = InputFile1.split('.')[0][:-1]

    img1 = cv2.imread(InputFile1, 0)
    img2 = cv2.imread(InputFile2, 0)

    ### Step1 : find correspondence across images
    sift = cv2.xfeatures2d.SIFT_create()
    kp1, des1 = sift.detectAndCompute(img1, None)
    kp2, des2 = sift.detectAndCompute(img2, None)

    BFmatch = BFMATCH(0.8, des1, des2, kp1, kp2)
    Mymatches, _ = BFmatch.B2M_30()
    x, xp = BFmatch.CorresspondenceAcrossImages()   # coordnate


    # turn to homogeneous coordinate
    h_x = np.ones( (x.shape[0], 3), dtype=float)
    h_xp = np.ones( (xp.shape[0], 3), dtype=float)
    h_x[:, :2] = x
    h_xp[:, :2] = xp
    h_x = h_x.T
    h_xp = h_xp.T

    ### Step2 : find fundamental matrix
    RSC8pt = RANSAC(thresh = 0.1, n_times = 1000, points = 50)
    F, idx = RSC8pt.ransac_8points(h_x, h_xp)

    ### Step3 : draw epipolar line
    inliers_x = h_x[:, idx]
    inliers_xp = h_xp[:, idx]

    lines_on_img1 = np.dot(F.T, inliers_xp).T
    lines_on_img2 = np.dot(F, inliers_x).T

    draw_epipolar(lines_on_img1, lines_on_img2, inliers_x, inliers_xp, img1, img2, filename)

    ### Step4 : get 4 possible essential matrix
    if case == "1":
        K1 = K2 = np.array([[1.4219, 0.0005, 0.5092],
                    [0, 1.4219, 0.3802],
                    [0, 0, 0.0010]], dtype=float)

    elif case == "2":
        K1 = np.array([[5426.566895, 0.678017, 330.096680],
                    [0.000000, 5423.133301, 648.950012],
                    [0.000000,    0.000000,   1.000000]], dtype=float)
        K2 = np.array([[5426.566895, 0.678017, 387.430023],
                    [0.000000, 5423.133301, 620.616699],
                    [0.000000,    0.000000,   1.000000]], dtype=float)

    else:
        K1 = K2 = np.array([[1065.08523, 0, 619.911384],
                    [0, 1063.81784, 511.949947],
                    [0, 0, 1]], dtype=float)

    m1, m2, m3, m4 = compute_essential(K1, K2, F)

    ### Step5 : find best essential matrix
    true_E, points3D = find_true_essential(m1, m2, m3, m4, inliers_x.T, inliers_xp.T)

    write_data(points3D, inliers_xp, true_E, filename)





