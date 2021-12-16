from glob import glob
import cv2
import os
import numpy as np
import random
from math import inf, sqrt
from collections import Counter
import copy
import cyvlfeat as vlfeat


def read_image(path):
    """
        Read images.
    """
    ret = []
    category = glob(os.path.join(path, "*"))
    category = sorted(category)
    for cls_id, cat_path in enumerate(category):
        img_names = glob(os.path.join(cat_path, "*"))
        for img_name in img_names:
            img = cv2.imread(img_name, 0)
            ret.append((img, cls_id))

    return ret


def get_features(dataset):
    """
        Args:
            dataset (list): Each element is a pair (image, class id).

        Returns:
            ret (np.ndarray): All feature descriptions in the dataset.
    """
    # sift = cv2.SIFT_create()
    ret = []
    for img, _ in dataset:
        # kp, des = sift.detectAndCompute(img, None)
        kps, des = vlfeat.sift.dsift(
            img, step=8, fast=True, float_descriptors=True)
        if len(kps) > 0:
            des = des[:int(len(des)*0.05)]
            ret.append(des)
    ret = np.concatenate(ret, axis=0)
    return ret


def preprocess(dataset, resize):
    """
        Resize and normalize.
    """
    # ret = [(cv2.resize(img, (16, 16)), cls_id)
    #        for img, cls_id in dataset]

    ret = []
    for img, cls_id in dataset:
        # resize
        if resize != -1:
            img = cv2.resize(img, (resize, resize))
        # normalize
        img = (img-np.mean(img))/np.std(img)
        img = img.astype(np.uint8)
        ret.append((img, cls_id))

    return ret


def distance(vec1, vec2):
    """
        Euclidean distance.
    """
    return sqrt(np.sum((vec1-vec2)**2))


def calculate_cluster_mean(cluster):
    """
        Calculate new cluster center.
    """
    ret = []
    for group in cluster:
        mean = np.mean(group, axis=0)
        ret.append(mean)
    ret = np.asarray(ret)
    return ret


def is_equal(cts1, cts2):
    """
        Check if centers and updated centers are all the same.
    """
    is_same = True
    for ct1, ct2 in zip(cts1, cts2):
        if not (ct1 == ct2).all():
            print(distance(ct1, ct2))
            # is_same = False
            return False

    return is_same


def clustering(des, k):
    """
        Separate all descriptions to k clusters using k-means.

        Args:
            des (list): All descriptions.
            k (int): K clusters.

        Returns:
            centers (list): K centers of k cluster.
            cluster (list): Each feature belongs to a bucket.
    """
    tmp_idx = random.sample(list(range(len(des))), k)
    new_centers = np.asarray([des[i] for i in tmp_idx])
    itr = 0
    while True:
        print(f"k means iter: {itr}")
        itr += 1
        centers = copy.deepcopy(new_centers)
        cluster = [[] for i in range(k)]
        for desc in des:
            min_dis = inf
            idx = -1
            for i, center in enumerate(centers):
                dis = distance(desc, center)
                if dis < min_dis:
                    min_dis = dis
                    idx = i
            cluster[idx].append(desc)
        new_centers = calculate_cluster_mean(cluster)
        if is_equal(new_centers, centers):
            break

    print(f"finish k-means clustering")
    return centers


def find_cloest_center(des, centers):
    """
        Find cloest center to the description.
    """

    min_dis = inf
    idx = -1
    for i, center in enumerate(centers):
        dis = distance(center, des)
        if dis < min_dis:
            min_dis = dis
            idx = i
    return idx


def build_histogram(dataset, centers):
    """
        Build histogram for every image in the dataset.
    """
    # sift = cv2.SIFT_create()
    ret = []
    for img, cls_id in dataset:
        hist = np.zeros(len(centers))
        # kp, desc = sift.detectAndCompute(img, None)
        kps, desc = vlfeat.sift.dsift(
            img, step=8, fast=True, float_descriptors=True)
        if len(kps) == 0:
            continue
        for des in desc:
            idx = find_cloest_center(des, centers)
            hist[idx] += 1
        ret.append((np.asarray(hist), cls_id))

    return ret


def knn(hist, train, k):
    """
        KNN algorithm.
    """
    dist = [(distance(hist, train_hist), train_cls)
            for train_hist, train_cls in train]
    dist = sorted(dist, key=lambda x: x[0])[:k]

    return Counter([cls_id for _, cls_id in dist]).most_common(1)[0][0]


def main():
    """
        Main
    """
    train_path = "hw5_data/train"
    test_path = "hw5_data/test"

    # read images
    train_set = read_image(train_path)
    test_set = read_image(test_path)

    # preprocess
    train_set = preprocess(train_set, resize=-1)
    test_set = preprocess(test_set, resize=-1)

    # get all feature points in train set
    des = get_features(train_set)

    k = 30
    # clustering using k-means
    centers = clustering(des, k=k)

    # build histogram
    train_hist = build_histogram(train_set, centers)

    # get features in test set
    test_hist = build_histogram(test_set, centers)

    # knn
    for knn_k in range(1, 11):
        correct = 0
        total_num = 0
        for hist, cls_id in test_hist:
            total_num += 1
            predict = knn(hist, train_hist, knn_k)
            if predict == cls_id:
                correct += 1
        result = (
            f"k: {knn_k:2d}, "
            f"total: {total_num}, "
            f"correct: {correct}, "
            f"ratio: {correct*100/total_num:.2f}%"
        )
        print(result)


if __name__ == "__main__":
    main()
