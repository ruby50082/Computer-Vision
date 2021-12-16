import numpy as np
from numpy.linalg import inv
import cv2

def WarPerspective(src, H, outshape):
    assert type(src) is np.ndarray, 'src is not np.ndarray'
    dst = np.zeros((outshape[0], outshape[1], src.shape[2]), dtype = np.uint8)
    for i in range(outshape[1]):
        for j in range(outshape[0]):
            p_trans = np.dot(H, [i,j,1])
            p_trans /= p_trans[2]
            x,y = int(p_trans[0]),int(p_trans[1])
            if x >= 0 and y >= 0 and x < src.shape[1] and y < src.shape[0]:
                dst[j,i] = src[y,x]
    cv2.imwrite('output/perspective.png', dst)
    return dst

def stitch(img_left, img_right, H):
    result = WarPerspective(img_right, H = H, outshape = (img_right.shape[0], img_right.shape[1]+img_left.shape[1]))
    border = img_left.shape[1]
    for i in range(img_left.shape[0]):
        for j in range(img_left.shape[1]):
            if any( val != 0 for val in result[i,j]):
                border = min(border, j)
    total_overlap = float(img_left.shape[1] - border)
    for i in range(img_left.shape[0]):
        for j in range(img_left.shape[1]):
            if any( val != 0 for val in result[i,j]):
                alpha = float(img_left.shape[1]-j)/total_overlap
                result[i,j] = (result[i,j]*(1-alpha) + img_left[i,j]*alpha).astype(int)
            else:
                result[i,j] = img_left[i,j]
    
    return result