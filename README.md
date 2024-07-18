# Mutual Improvement of 3D Human Pose and Mesh Estimation by Leveraging Anthropometric Measurements for Consistent Body Shapes

## Installation

Create a new anaconda environment. By running 
```bash
conda env create -f environment.yml
```
and activate the environment with
```bash
conda activate a2b_hme
```
Further, download the SMPL-X models from the [official SMPL-X repository](https://smpl-x.is.tue.mpg.de) and place them in the folder `smpl-models/smplx`. Alternatively, you can modify the variable `SMPL_MODEL_DIR` in the file `config.py` and set it to the directory where you have stored the model files.

## A2B Models

We provide the inference code for our A2B models. All anthropometric measurements should be given in a json file. This file needs to contain a dicionary mapping from subject names to anthropometric measurements. The measurements need to be given in a dictionary mapping from measurement key to the value in meters. The following measurements are expected:

`height length, shoulder width length, torso height from back length, torso height from front length, head length, midline neck length, lateral neck length, left hand length, right hand length, left arm length, right arm length, left forearm length, right forearm length, left thigh length, right thigh length, left calf length, right calf length, left footwidth length, right footwidth length, left heel to ball length, right heel to ball length, left heel to toe length, right heel to toe length, waist circumference, chest circumference, hip circumference, head circumference, neck circumference, left arm circumference, right arm circumference, left forearm circumference, right forearm circumference, left thigh circumference, right thigh circumference, left calf circumference, right calf circumference`

The landmarks for the measurements are given in the supplementary material of the paper. In order to run the inference, execute

```bash
python -m anthro.inference --measurements <path_to_measurements_json> --save_path <path_to_save_beta_predictions> --name <measurement_name>
```

We provide an example anthropometric measurements file `anthro/example_measurements.json` in order to test our models.
```bash
python -m anthro.inference --measurements "./anthro/example_measurements.json" --save_path "./anthro/anthro_betas.json" --name example
```

## Uplift Upsample
This repository contains a reimplementation
of the Tensorflow code from [Uplift and Upsample: Efficient 3D Human Pose Estimation with Uplifting Transformers](https://arxiv.org/abs/2210.06110) in PyTorch. You can find all the code in the subdirectory `uplift_upsample`.

Follow the instructions of the [original repository](https://github.com/goldbricklemon/uplift-upsample-3dhpe) to setup the dataset(s), download the pre-trained models, and run the code. Since the code is a sub-repository here, the only difference for running is that you need to add the `uplift_upsample` prefix to the train/eval calls, etc. Pretrained weights can be downloaded from the original repository in `.h5` format. This repository provides a converter from Tensorflow weights in `h5` format to PyTorch weights. Put the downloaded pretrained weights in the following folder: `./uplift_upsample/pretrained_weights`.

We further add experiments and data preparation code for ASPset and fit3D. In order to use ASPset, you need to install the code provided by the [official repository](https://github.com/anibali/aspset-510). Both fit3d and ASPset expect 2D predictions from ViTPose as an input. These files are expected as csv files for each split. We provide the files for ASPset in the directory `dataset/aspset/vitpose`. The files for fit3D will be uploaded after acceptance of this paper due to file size restrictions and anonymity requirements. You can train uplift upsample on ASPset with the following call:

```bash
python -m uplift_upsample.train --config ./uplift_upsample/experiments/aspset_351.json --out_dir ./uplift_upsample/out --dataset_3d_path <path/to/aspset> --train_subset train --val_subset val --dataset aspset --gpu_id 0
```