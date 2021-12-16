# Introduction
- The original approach in Homework 5, BoW (Bag of Words), simply counts the number of descriptors associated with each cluster in a codebook, and creates a histogram for each set of descriptors from an image.
- Therefore, BoW doesn’t consider much information, such as the first order statistics and the relevance among local descriptors in an image. We would like to try different approaches in order to improve this part and in turn perform the classification with SVM.
- Second, we try a deep learning method, named NetVLAD, which is built on top of the VLAD algorithm with NN architecture. It can not only retain the advantages of the classical VLAD algorithm, but also use neural networks to learn better parameters. Thus, NetVLAD is expected to generate better descriptors than the VLAD.
- Third, we also further train two end-to-end networks, VGG16 and ResNet34 to perform classification.
- More details can be found in report.
