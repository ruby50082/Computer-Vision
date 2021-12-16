import cv2
import numpy as np

class RANSAC():
    def __init__(self, thresh, n_times, points):
        self.thresh = thresh
        self.n_times = n_times
        self.points = points
    
    def Cal8points_err(self, x1, x2, F):
        Fx1 = np.dot(F, x1)
        Fx2 = np.dot(F, x2)
        denom = Fx1[0]**2 + Fx1[1]**2 + Fx2[0]**2 + Fx2[1]**2
        test_err = (np.diag(np.dot(np.dot(x1.T, F), x2)))**2 / denom

        return test_err
    
    def ransac_8points(self, x1, x2):

        Ans_F = None
        max_inlier = []
        npts = x1.shape[1]

        for iter_1 in range(self.n_times):
            all_idxs = np.arange(npts)
            np.random.shuffle(all_idxs)
            try_idxs = all_idxs[:self.points]
            test_idxs = all_idxs[self.points:]
            try_x1 = x1[:, try_idxs]
            try_x2 = x2[:, try_idxs]
            test_x1 = x1[:, test_idxs]
            test_x2 = x2[:, test_idxs]
            
            maybe_F, _ = cv2.findFundamentalMat(try_x1.T, try_x2.T, cv2.FM_LMEDS)
            # maybe_F = self.fundamentalFit(try_x1.T, try_x2.T)
            test_err = self.Cal8points_err(test_x1, test_x2, maybe_F)

            now_inlier = list(try_idxs)
            for iter_err in range(len(test_err)): 
                if test_err[iter_err] < self.thresh:
                    now_inlier.append(test_idxs[iter_err])
            
            if len(now_inlier) > len(max_inlier):
                Ans_F = maybe_F
                max_inlier = now_inlier
        
        if Ans_F is None:
            raise ValueError("didn't find F")
        
        return Ans_F, max_inlier

    def fundamentalFit(self, p1, p2):

        na,Ta = self.normalizeHomogeneous(p1)
        nb,Tb = self.normalizeHomogeneous(p2)

        p2x1p1x1 = nb[:,0] * na[:,0]
        p2x1p1x2 = nb[:,0] * na[:,1]
        p2x1 = nb[:, 0]
        p2x2p1x1 = nb[:,1] * na[:,0]
        p2x2p1x2 = nb[:,1] * na[:,1]
        p2x2 = nb[:,1]
        p1x1 = na[:,0]
        p1x2 = na[:,1]
        ones = np.ones((1,p1.shape[0]))

        A = np.vstack([p2x1p1x1,p2x1p1x2,p2x1,p2x2p1x1,p2x2p1x2,p2x2,p1x1,p1x2,ones])
        A = np.transpose(A)

        u, D, v = np.linalg.svd(A)
        vt = v.T
        F = vt[:, 8].reshape(3,3)

        u, D, v = np.linalg.svd(F)
        F=np.dot(np.dot(u, np.diag([D[0], D[1], 0])), v)
        F= np.dot(np.dot(Tb,F),np.transpose(Ta))

        return F

    def normalizeHomogeneous(self, points):
        if points.shape[1] == 2:
            points = np.hstack([points, np.ones((points.shape[0],1))])

        n = points.shape[0]
        d = points.shape[1]
        factores = np.repeat((points[:, -1].reshape(n, 1)), d, axis=1)
        points = points / factores

        prom = np.mean(points[:,:-1],axis=0)
        newP = np.zeros(points.shape)
        newP[:,:-1] = points[:,:-1] - np.vstack([prom for i in range(n)])

        dist = np.sqrt(np.sum(np.power(newP[:,:-1],2),axis=1))
        meanDis = np.mean(dist)
        scale = np.sqrt(2)*1.0/ meanDis

        T = [[scale,0,-scale*prom[0]],
            [0, scale, -scale * prom[1]],
            [0, 0, 1]]

        T = np.transpose(np.array(T))
        transformedPoints = np.dot(points,T)

        return transformedPoints, T
