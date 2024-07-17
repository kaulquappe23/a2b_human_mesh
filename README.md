# Mutual Improvement of 3D Human Pose and Mesh Estimation by Leveraging Anthropometric Measurements for Consistent Body Shapes

## Installation

## A2B Models

We provide the inference code for our A2B models. All anthropometric measurements should be given in a json file. This file needs to contain a dicionary mapping from subject names to anthropometric measurements. The measurements need to be given in a dictionary mapping from measurement key to the value in meters. The following measurements are expected: `height length, shoulder width length, torso height from back length, torso height from front length, head length, midline neck length, lateral neck length, left hand length, right hand length, left arm length, right arm length, left forearm length, right forearm length, left thigh length, right thigh length, left calf length, right calf length, left footwidth length, right footwidth length, left heel to ball length, right heel to ball length, left heel to toe length, right heel to toe length, waist circumference, chest circumference, hip circumference, head circumference, neck circumference, left arm circumference, right arm circumference, left forearm circumference, right forearm circumference, left thigh circumference, right thigh circumference, left calf circumference, right calf circumference`. The landmarks for the measurements are given in the supplementary material of the paper. In order to run the inference, execute

```bash
python anthro/inference.py --measurements <path_to_measurements_json> --save_path <path_to_save_beta_predictions> --name <measurement_name>
```


## Uplift Upsample
This repository contains a reimplementation
of the Tensorflow [Uplift and Upsample: Efficient 3D Human Pose Estimation with Uplifting Transformers](https://arxiv.org/abs/2210.06110) code in PyTorch. You can find all the code in the subdirectory `uplift_upsample`.

Follow the instructions of the [original repository](https://github.com/goldbricklemon/uplift-upsample-3dhpe) to setup the dataset(s), download the pre-trained models, and run the code. This repository provides a converter from Tensorflow to PyTorch weights. Since the code is a sub-repository here, the only difference is that you need to add the `uplift_upsample` prefix to the train/eval calls, etc.  

We further add experiments and data preparation code for ASPset and fit3D. In order to use ASPset, you need to install the code provided by the [official repository](https://github.com/anibali/aspset-510). Both fit3d and ASPset expect 2D predictions from ViTPose as an input. These files are expected as csv files for each split. We provide the files for ASPset in the directory `dataset/aspset/vitpose`. The files for fit3D will be uploaded after acceptance of this paper due to file size restrictions and anonymity requirements.
