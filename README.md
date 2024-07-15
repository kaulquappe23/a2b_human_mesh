# Mutual Improvement of 3D Human Pose and Mesh Estimation by Leveraging Anthropometric Measurements for Consistent Body Shapes

## Uplift Upsample
This repository contains a reimplementation
of the Tensorflow [Uplift and Upsample: Efficient 3D Human Pose Estimation with Uplifting Transformers](https://arxiv.org/abs/2210.06110) code in PyTorch. You can find all the code in the subdirectory `uplift_upsample`.

You can follow the instructions of the [original repository](https://github.com/goldbricklemon/uplift-upsample-3dhpe) to setup the dataset(s) and download the pre-trained models. This repository provides a converter from Tensorflow to PyTorch weights. Since the code is a sub-repository here, the only difference is that you need to add the `uplift_upsample` prefix to the train/eval calls, etc.

We further add experiments and data preparation code for ASPset and fit3D. In order to use ASPset, you need to install the code provided by the [official repository](https://github.com/anibali/aspset-510). Both fit3d and ASPset expect 2D predictions from ViTPose as an input. These files are expected as csv files for each split.  