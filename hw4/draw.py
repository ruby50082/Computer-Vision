import cv2
import numpy as np

def draw_epipolar(l, lp, x, xp, img1, img2, filename):
    pic1 = cv2.cvtColor(img1, cv2.COLOR_GRAY2BGR)
    pic2 = cv2.cvtColor(img2, cv2.COLOR_GRAY2BGR)
    h, w = img1.shape
    
    for r, pt_x, pt_xp in zip(l, x.T, xp.T):
        color = tuple(np.random.randint(0, 255, 3).tolist())
        x0 = 0
        y0 = (-r[2]/r[1]).astype(np.int)
        x1 = w
        y1 = (-(r[2]+r[0]*w)/r[1]).astype(np.int)
        pic1 = cv2.line(pic1, (x0, y0), (x1, y1), color, 1)

        pic1 = cv2.circle(pic1, tuple((int(pt_x[0]), int(pt_x[1]))), 3, color, -1)
        pic2 = cv2.circle(pic2, tuple((int(pt_xp[0]), int(pt_xp[1]))), 3, color, -1)
        
    cv2.imwrite('./output/' + filename + '_left.png', np.concatenate((pic1, pic2), axis=1))

    pic1 = cv2.cvtColor(img1, cv2.COLOR_GRAY2BGR)
    pic2 = cv2.cvtColor(img2, cv2.COLOR_GRAY2BGR)

    for r, pt_x, pt_xp in zip(lp, x.T, xp.T):
        color = tuple(np.random.randint(0, 255, 3).tolist())
        x0 = 0
        y0 = (-r[2]/r[1]).astype(np.int)
        x1 = w
        y1 = (-(r[2]+r[0]*w)/r[1]).astype(np.int)
        pic2 = cv2.line(pic2, (x0,y0), (x1,y1), color, 1)
        pic1 = cv2.circle(pic1, tuple((int(pt_x[0]), int(pt_x[1]))), 3, color, -1)
        pic2 = cv2.circle(pic2, tuple((int(pt_xp[0]), int(pt_xp[1]))), 3, color, -1)

    cv2.imwrite('./output/' + filename + '_right.png', np.concatenate((pic1, pic2), axis=1))


def write_data(points3D, inliers_xp, true_E, filename):
    fp = open('output/plot_' + filename + ".txt", "w")
    
    fp.write('3d\n\n')
    fp.write(str(points3D.T))
    
    fp.write('\n2d\n\n')
    xpp = inliers_xp.T[:, :2]
    fp.write(str(xpp))

    fp.write('\nessential\n')
    fp.write(str(true_E))
    fp.close()  