# Introduction
- Image stitching is the process of combining multiple images with overlapping field to produce a panorama image.
- Before blending the images, we need to align them first.
- Using the keypoints and descriptors calculated by SIFT, we would know the correspondences between the two images.
- Finally, we use RANSAC algorithm to estimate a homography matrix and apply warping transformation to stitch the two images.
- <img src=https://user-images.githubusercontent.com/39916963/146356020-96e9d5ac-bfad-4332-8fb0-c76c7636d6c1.png width="40%" height="40%">
- The details can be found in report.
