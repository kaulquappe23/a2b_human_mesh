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
For evaluation and fit3D, the SMPL-X models are needed. If you just want to test the A2B models, the models are not necessary. If you need them, download the SMPL-X models from the [official SMPL-X repository](https://smpl-x.is.tue.mpg.de) and place them in the folder `smpl-models/smplx`. Alternatively, you can modify the variable `SMPL_MODEL_DIR` in the file `config.py` and set it to the directory where you have stored the model files.

## A2B Models

We provide the inference code for our A2B models. All anthropometric measurements should be given in a json file. This file needs to contain a dictionary mapping from subject names to anthropometric measurements. The measurements need to be given in a dictionary mapping from measurement key to the value in meters. The following measurements are expected:

`height length, shoulder width length, torso height from back length, torso height from front length, head length, midline neck length, lateral neck length, left hand length, right hand length, left arm length, right arm length, left forearm length, right forearm length, left thigh length, right thigh length, left calf length, right calf length, left footwidth length, right footwidth length, left heel to ball length, right heel to ball length, left heel to toe length, right heel to toe length, waist circumference, chest circumference, hip circumference, head circumference, neck circumference, left arm circumference, right arm circumference, left forearm circumference, right forearm circumference, left thigh circumference, right thigh circumference, left calf circumference, right calf circumference`

Therefore, the format of the json input file should look like the following. We further provide an example anthropometric measurements file `anthro/example_measurements.json`. The landmarks used for the measurements are given in the supplementary material of the paper.
```json
{
    "subject1": {
        "height length": 1.8,
        "shoulder width length": 0.4,
        "torso height from back length": 0.4,
        "torso height from front length": 0.4,
        "head length": 0.2,
        "midline neck length": 0.1,
        "lateral neck length": 0.1,
        "left hand length": 0.2,
        "right hand length": 0.2,
        "left arm length": 0.3,
        "right arm length": 0.3,
        "left forearm length": 0.2,
        "right forearm length": 0.2,
        "left thigh length": 0.5,
        "right thigh length": 0.5,
        "left calf length": 0.4,
        "right calf length": 0.4,
        "left footwidth length": 0.1,
        "right footwidth length": 0.1,
        "left heel to ball length": 0.2,
        "right heel to ball length": 0.2,
        "left heel to toe length": 0.3,
        "right heel to toe length": 0.3,
        "waist circumference": 0.8,
        "chest circumference": 1.0,
        "hip circumference": 0.9,
        "head circumference": 0.5,
        "neck circumference": 0.3,
        "left arm circumference": 0.2,
        "right arm circumference": 0.2,
        "left forearm circumference": 0.2,
        "right forearm circumference": 0.2,
        "left thigh circumference": 0.3,
        "right thigh circumference": 0.3,
        "left calf circumference": 0.2,
        "right calf circumference": 0.2
    },
    "subject2": {
        ...
    }
}
```
 In order to run the inference, execute

```bash
python -m anthro.inference --measurements <path_to_measurements_json> --save_path <path_to_save_beta_predictions> --name <measurement_name>
```

You can try the inference with the example file by executing:
```bash
python -m anthro.inference --measurements "./anthro/example_measurements.json" --save_path "./anthro/anthro_betas.json" --name example
```

The output then looks like the following:

```json
{
    "example_A2B_svr_male":
    {
        "gender": "male",
        "name": "example_A2B_svr_male",
        "subject_1":
        [
            1.5561325028191986,
            -0.8166642549933809,
            1.0529247639574861,
            -0.9301149961670858,
            -1.2445880571259105,
            0.3664527874039336,
            0.28154321636613916,
            3.45716809571207,
            -0.5160723521828383,
            -1.795204227512637,
            2.1629444945670566
        ],
        "subject_2": ...,
        "num_betas": 11
    },
    "example_A2B_nn_male":
    {
        "gender": "male",
        "name": "example_A2B_nn_male",
        "subject_1":
        [
            1.4699337482452393,
            -0.9374016523361206,
            0.8422941565513611,
            -0.5198159217834473,
            -5.681121349334717,
            -1.4253689050674438,
            0.390656977891922,
            2.2848150730133057,
            -0.3818075656890869,
            -1.983944296836853,
            1.1676679849624634
        ],
        "subject_2": ...,
        "num_betas": 11
    },
    "example_A2B_svr_neutral":...,
    "example_A2B_nn_neutral":...,
    "example_A2B_svr_female":...,
  

}
```
The output is given per evaluated A2B model (SVR, NN) and per gender (neutral, female, male). For each model output, you will find the name (of model and anthropometric measurements as given on the command line), the gender, the number of beta parameters used and the results per subject. 

## Inverse Kinematics

We provide the code to run inverse kinematics (IK) on the fit3D ground truth data. Running it on any other set of 3D poses can be added straightforward. The code is provided in the subdirectory `inverse_kinematics`. You need vposer to run the code. Download the pretrained vposer model from the [official VPoser repository](https://smpl-x.is.tue.mpg.de) and place it in the folder `inverse_kinematics/V02_05`. Alternatively, you can modify the variable `VPOSER_DIR` in the file `config.py` and set it to the directory where you have stored the model files. To run IK on the fit3D ground truth, execute the following command:

```bash
python -m inverse_kinematics.run_ik --data_path <path/to/fit3d> --save_path <path_to_save_ik_results> --gpus <"list of gpus"> --num_procs <number of processes> --gender <gender_of_smplx_model> --split <val or train>
```

## Uplift Upsample
This repository contains a reimplementation
of the Tensorflow code from [Uplift and Upsample: Efficient 3D Human Pose Estimation with Uplifting Transformers](https://arxiv.org/abs/2210.06110) in PyTorch. You can find all the code in the subdirectory `uplift_upsample`.

### Training on ASPset
In order to train Uplift Upsample on ASPset, you need to download the dataset with the code provided by the [official  ASPset repository](https://github.com/anibali/aspset-510). Training on ASPset requires 2D predictions from ViTPose as an input. These files are expected as csv files for each split. We provide the files for ASPset in the directory `dataset/aspset/vitpose`. We further use the pretrained weights from the AMASS pretrained and Human3.6m finetuned variant. Therefore, download `h36m_351_pt.h5` from the [official Uplift Upsample repository](https://github.com/goldbricklemon/uplift-upsample-3dhpe). Our repository provides a converter from their Tensorflow weights in `h5` format to PyTorch weights. Put the downloaded pretrained weights in the following folder: `./uplift_upsample/pretrained_weights`.

You can train uplift upsample on ASPset with the following call. Replace `dataset_3d_path` with the path where you stored the ASPset files.
```bash
python -m uplift_upsample.train --config ./uplift_upsample/experiments/aspset_351.json --out_dir ./uplift_upsample/out --dataset_3d_path <path/to/aspset> --train_subset train --val_subset val --dataset aspset --gpu_id 0
```

### Training on fit3D
Training on fit3D is executed analogous to ASPset. The csv files for fit3D will be uploaded after acceptance of this paper due to file size restrictions and anonymity requirements. 

### Training on AMASS and Human3.6m
Follow the instructions of the [original repository](https://github.com/goldbricklemon/uplift-upsample-3dhpe) to setup the dataset(s), download the pre-trained models, and run the code. Since the code is a sub-repository here, the only difference for running is that you need to add the `uplift_upsample` prefix to the train/eval calls, etc. Pretrained weights can be downloaded from the original repository in `.h5` format and reused in this repository.

## The Rest of the Code
All other functionality (IK, Weights, B2A, Evaluation, Regressors, etc.) will be added after acceptance of this paper.