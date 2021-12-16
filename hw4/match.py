import cv2
import math
import numpy as np

class BFMATCH():
    def __init__(self, thresh, des1, des2, kp1, kp2):
        self.thresh = thresh
        self.match = []
        self.asm = []
        self.x = []
        self.xp = []
        self.des1 = des1
        self.des2 = des2
        self.kp1 = kp1
        self.kp2 = kp2

    def Best2Matches(self):
        idx1 = 0
        mmatch = []
        for p1 in self.des1:
            best_m = []
            temp0 = cv2.DMatch(idx1, 0, math.sqrt((p1 - self.des2[0]).T.dot(p1 - self.des2[0])))
            temp1 = cv2.DMatch(idx1, 1, math.sqrt((p1 - self.des2[1]).T.dot(p1 - self.des2[1])))
            if temp0.distance < temp1.distance:
                best_m.append(temp0)
                best_m.append(temp1)       
            else:
                best_m.append(temp1)
                best_m.append(temp0)

            idx2 = 0
            for p2 in self.des2:
                dis = math.sqrt((p1-p2).T.dot((p1-p2)))
                if dis < best_m[0].distance:
                    best_m[0].trainIdx = idx2
                    best_m[0].distance = dis
                elif dis < best_m[1].distance:
                    best_m[1].trainIdx = idx2
                    best_m[1].distance = dis
                idx2 = idx2 + 1
            idx1 = idx1 + 1
            mmatch.append(best_m)   
        return mmatch

    def B2M_30(self):
        temp = []
        MATCH = []
        self.match  = self.Best2Matches()
        for m in self.match:
            if m[0].distance < (self.thresh * m[1].distance):
                temp.append((m[0].trainIdx, m[0].queryIdx))
                MATCH.append(m[0])
        Mymatches = np.asarray(temp)
        MATCH = sorted(MATCH, key=lambda x: x.distance)
        thirty_match = MATCH[:30]
        
        temp30 =[]
        for dmatch in thirty_match:
            temp30.append((dmatch.trainIdx, dmatch.queryIdx))
        thirty_match = np.asarray(temp30)
        return Mymatches, thirty_match
            
    def CorresspondenceAcrossImages(self):
        for i, (m, n) in enumerate(self.match):
            if m.distance < self.thresh * n.distance:
                self.asm.append(m)
                self.x.append(self.kp1[m.queryIdx].pt)
                self.xp.append(self.kp2[m.trainIdx].pt)

        self.x = np.asarray(self.x)
        self.xp = np.asarray(self.xp)
        return self.x, self.xp


