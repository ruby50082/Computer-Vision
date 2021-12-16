import cv2
from glob import glob
import os
from math import sqrt
from collections import Counter
import numpy as np
def read_image(path):
    ret = []
    category = glob(os.path.join(path, "*"))
    category = sorted(category)
    for cls_id, cat_path in enumerate(category):
        img_names = glob(os.path.join(cat_path, "*"))
        for img_name in img_names:
            img = cv2.imread(img_name, 0)
            ret.append((img, cls_id))

    return ret


def preprocess(dataset):
    ret = []
    for img, cls_id in dataset:
        # resize
        img = cv2.resize(img, (16, 16))
        # normalize
        img = (img-np.mean(img))/np.std(img)
        ret.append((img, cls_id))

    return ret


def distance(img1, img2):
    return sqrt(np.sum((img1-img2)**2))


def knn(test, train, k):
    # calculate image distance between test image and all train image
    dist = [(distance(train_img, test), train_cls)
            for train_img, train_cls in train]
    dist = sorted(dist, key=lambda x: x[0])[:k]

    # find cloest class
    return Counter([cls_id for _, cls_id in dist]).most_common(1)[0][0]


def main():
    train_path = "hw5_data/train"
    test_path = "hw5_data/test"
    
    # read images
    train_set = read_image(train_path)
    test_set = read_image(test_path)

    # preprocess
    train_set = preprocess(train_set)
    test_set = preprocess(test_set)

    # knn
    for k in range(1, 11):
        correct = 0
        total_num = 0
        for img, cls_id in test_set:
            total_num += 1
            predict = knn(img, train_set, k)
            if predict == cls_id:
                correct += 1
        result = (
            f"k: {k:2d}, "
            f"total: {total_num}, "
            f"correct: {correct}, "
            f"ratio: {correct*100/total_num:.2f}%"
        )
        print(result)

if __name__ == "__main__":
    main()
