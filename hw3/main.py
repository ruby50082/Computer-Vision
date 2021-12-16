import cv2
import os
import numpy as np
import random
from warp import stitch


in_path = 'data'
out_path = 'output'
def find_match(des1, des2):
    ret = []
    dist_thresh = 0.6
    for i, des_1 in enumerate(des1):
        best = [1e6, 1e6]
        best_idx = [-1, -1]
        for j, des_2 in enumerate(des2):
            dist = ssd(des_1, des_2)
            if dist < best[0]:
                best[0], best[1] = dist, best[0]
                best_idx[0], best_idx[1] = j, best_idx[0]
            elif dist < best[1]:
                best[1] = dist
                best_idx[1] = j
        if L2_dist(des_1, des2[best_idx[0]])/L2_dist(des_1, des2[best_idx[1]])\
           < dist_thresh:
            ret.append([i, best_idx[0]])
    
    return ret


def L2_dist(des1, des2):
    return np.sqrt(ssd(des1, des2))


def ssd(des1, des2):
    return sum((des1-des2)**2)


def draw_match_lines(img1, img2, kp1, kp2, matches):
    kp1 = np.asarray(kp1)
    kp2 = np.asarray(kp2)
    kp1 = kp1[matches[:, 0]]
    kp2 = kp2[matches[:, 1]]
    img = cv2.hconcat([img1, img2])
    for p1, p2 in zip(kp1[:20], kp2[:20]):
        #print(p1.pt[0], p1.pt[1])
        cv2.line(img, (int(p1.pt[0]), int(p1.pt[1])), (int(
            p2.pt[0])+img1.shape[1], int(p2.pt[1])), (0, 0, 255), 1)
    cv2.imwrite(os.path.join(out_path, 'match.jpg'), img)
    

def homography(pnt1, pnt2):
    
    pnt1 = [[kp.pt[0], kp.pt[1], 0] for kp in pnt1]
    pnt2 = [[kp.pt[0], kp.pt[1]] for kp in pnt2]
    pnt1 = np.asarray(pnt1)
    pnt2 = np.asarray(pnt2)

    P = np.empty([0, 9])
    for pt_idx in range(len(pnt1)):
        x = pnt1[pt_idx, 0]
        y = pnt1[pt_idx, 1]
        u = pnt2[pt_idx, 0]
        v = pnt2[pt_idx, 1]
        arr1 = np.array([x, y, 1, 0, 0, 0, -u*x, -u*y, -u])
        arr2 = np.array([0, 0, 0, x, y, 1, -v*x, -v*y, -v])
        P = np.concatenate(
            (P, arr1[np.newaxis, :], arr2[np.newaxis, :],), axis=0)
    U, S, Vt = np.linalg.svd(P)
    H = Vt[-1]
    H /= H[-1]
    H = np.reshape(H, (3, 3))
    # pnt1 = pnt1[:, :2]
    # builtin_h, status = cv2.findHomography(pnt1, pnt2)

    return H


def count_out_liers(H, kp1, kp2):
    cnt = 0
    threshold = 2
    for i, kp in enumerate(kp1):
        p = np.array([kp.pt[0], kp.pt[1], 1])
        warp_p = H.dot(p)
        warp_p = (warp_p / warp_p[-1])[:2]
        dist = L2_dist(warp_p, kp2[i].pt)
        #print(dist)
        if dist > threshold:
            cnt += 1

    return cnt


def ransac(kp1, kp2, matches):
    n_sample = 8
    outlier_ratio = 0.05
    n_iter = int(np.log(1 - 0.99) / np.log(1 - (1-outlier_ratio)**n_sample))
    best_homo = None
    best = 1e6
    print("iter: ", n_iter)
    kp1 = np.asarray(kp1)
    kp2 = np.asarray(kp2)
    kp1 = kp1[matches[:, 0]]
    kp2 = kp2[matches[:, 1]]
    print("matches: ", len(matches))
    # for _ in range(n_iter):
    for _ in range(20):
        # random sample matches feature points
        samples = random.sample(range(len(kp1)), n_sample)
        samples = np.asarray(samples)
        # calculate homography
        tmp_homo = homography(kp1[samples], kp2[samples])
        # calculate in/out liers
        outl_cnt = count_out_liers(tmp_homo, kp1, kp2)
        print(outl_cnt)
        if outl_cnt < best:
            best = outl_cnt
            best_homo = tmp_homo
    
    return best_homo


def main():
    if not os.path.isdir(out_path):
        os.makedirs(out_path)
    img1_name = 'hill1.JPG'
    img2_name = 'hill2.JPG'
    img1 = cv2.imread(os.path.join(in_path, img1_name))
    img2 = cv2.imread(os.path.join(in_path, img2_name))
    img1_gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    img2_gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    
    # Step1: get sift keypoints and descriptions
    sift = cv2.xfeatures2d.SIFT_create()
    print('creat descriptors')
    kp1, des1 = sift.detectAndCompute(img1_gray, None)
    kp2, des2 = sift.detectAndCompute(img2_gray, None)
    # img1 = cv2.drawKeypoints(img1_gray, kp1, img1)
    # img2 = cv2.drawKeypoints(img2_gray, kp2, img2)

    # cv2.imwrite(os.path.join(out_path, img1_name), img1)
    # cv2.imwrite(os.path.join(out_path, img2_name), img2)

    # Step2: feature matching
    # bf = cv2.BFMatcher()
    # matches = bf.match(des1, des2)
    # matches = sorted(matches, key=lambda x: x.distance)
    # img3 = None
    # img3 = cv2.drawMatches(img1, kp1, img2, kp2, matches[:20], img3,
    #                        flags=2)
    # cv2.imwrite(os.path.join(out_path, "match.jpg"), img3)
    print('find matches')
    matches = find_match(des1, des2)
    matches = np.asarray(matches)
    # print(matches)
    draw_match_lines(img1, img2, kp1, kp2, matches)

    # Step3: RANSAC
    print('ransac')
    homo = ransac(kp1, kp2, matches)

    # Step4: Warp image and stitch
    print('warp and stitch')
    stitch_img = stitch(img1, img2, homo)
    cv2.imwrite(os.path.join(out_path, 'result.png'), stitch_img)



    


if __name__ == "__main__":
    main()
