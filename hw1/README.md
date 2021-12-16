# Introduction
- Implementation camera calibration from scratch without using cv2.calibrateCamera or other calibration functions.
- <img src="https://user-images.githubusercontent.com/39916963/146325301-2f0652de-3345-4fac-9dee-129ca535b088.png" width="50%" height="50%">
- Camera calibration:
  - First, figure out the Hi of each images.
  - Use Hi to find B, and calculate intrinsic matrix K from B by using Cholesky factorization.
  - Then, get extrinsic matrix [R|t] for each images by K and H 
