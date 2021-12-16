import numpy as np

def compute_essential(K1, K2, F):
    E = np.dot(np.dot(K1.T, F), K2)

    U, D, V = np.linalg.svd(E)
    e = (D[0] + D[1]) / 2
    D[0] = D[1] = e
    D[2] = 0
    E = np.dot(np.dot(U, np.diag(D)), V)
    U, D, V = np.linalg.svd(E)
    W = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])
    
    R1 = np.dot(np.dot(U, W), V.T)
    R2 = np.dot(np.dot(U, W.T), V.T)
    if np.linalg.det(V) < 0:
        V = -V 
    if np.linalg.det(R2) < 0:
        U = -U
    U3 = U[:, -1]

    m1 = np.vstack((R1.T, U3)).T
    m2 = np.vstack((R1.T, -U3)).T
    m3 = np.vstack((R2.T, U3)).T
    m4 = np.vstack((R2.T, -U3)).T
    
    return m1, m2, m3 ,m4

def find_true_essential(m1, m2, m3, m4, x, xp):
    pt1 = triangular(m1, x, xp)
    pt2 = triangular(m2, x, xp)
    pt3 = triangular(m3, x, xp)
    pt4 = triangular(m4, x, xp)
    
    count1 = count_p_front(pt1, m1)
    count2 = count_p_front(pt2, m2)
    count3 = count_p_front(pt3, m3)
    count4 = count_p_front(pt4, m4)
    ansE = m1
    max_count = count1
    anspt = pt1
    if count2 > max_count:
        ansE = m2
        max_count = count2
        anspt = pt2
    if count3 > max_count:
        ansE = m3
        max_count = count3
        anspt = pt3
    if count4 > max_count:
        ansE = m4
        max_count = count4     
        anspt = pt4

    return ansE, anspt

def triangular(P2, x, xp):
    P1 = np.array([[1, 0, 0, 0],
                   [0, 1, 0, 0],
                   [0, 0, 1, 0]], dtype=float)
    
    p1 = P1[0, :]
    p2 = P1[1, :]
    p3 = P1[2, :]
    
    pp1 = P2[0, :]
    pp2 = P2[1, :]
    pp3 = P2[2, :]
    
    pointsX =[]
    for p, pp in zip(x, xp):
        u = p[0]
        v = p[1]
        up = pp[0]
        vp = pp[1]
        
        A = np.array([u*p3.T - p1.T,
                      v*p3.T - p2.T,
                      up*pp3.T - pp1.T,
                      vp*pp3.T - pp2.T])
        
        U, S, V = np.linalg.svd(A)
        X = V[:, -1]
        pointsX.append(X)
    pointsX = np.array(pointsX)
    
    for i in range(pointsX.shape[1]-1):
        pointsX[:,i] = pointsX[:,i] / pointsX[:,3]
    pointsX = pointsX[:,:3].T
    return pointsX

def count_p_front(points, m):

    camera_c = np.dot(m[:, 0:3], m[:, 3].T)
    count = 0
    for pt in points.T:
        if np.dot((pt - camera_c), m[:, 2].T) > 0:
            count = count + 1

    return count    
        
