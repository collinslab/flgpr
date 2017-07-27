# Forward-looking ground-penetrating radar (FLGPR) image feature processing package

`version 1.0`

This repo is created to house the code used to create many of the resutls found in a [recent feature comparison paper](https://arxiv.org/abs/1702.03000).

This package contains the following feature set creation abilities:
- image normalization
- scale invariant feature tranform (SIFT) descriptor
- local stiatics (LSTAT)
- 2D FFT
- Log-Gabor statistical feature
- Bag-of-visual words (BOV)
- Fisher vector(FV)

The code here depends on two other packages:
- [PRT -- Pattern Recogntion Toolbox](https://github.com/covartech/PRT)
- [VLFeat -- Vision Lab Features Library](https://github.com/vlfeat/vlfeat)