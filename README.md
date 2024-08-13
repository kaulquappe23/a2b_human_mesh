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

We provide the code to run inverse kinematics (IK) on the ASPset as well as fit3D ground truth data and results from the Uplift Upsample model. Running it on any other set of 3D poses can be added analogously. The code is provided in the subdirectory `inverse_kinematics`. You need vposer to run the code. Download the pretrained vposer model from the [official VPoser repository](https://smpl-x.is.tue.mpg.de) and place it in the folder `inverse_kinematics/V02_05`. Alternatively, you can modify the variable `VPOSER_DIR` in the file `config.py` and set it to the directory where you have stored the model files. 

The results are stored per video. A npz-file is generated per video, containing the results as a dictionary with the SMPL-X parameters (`trans`, `root_orient`, `pose_body`, `betas`) and the values per video frame. Before executing IK, the poses are converted such that the pelvis is in the origin. This speeds up the process. In order to convert the poses back to the original coordinate system, the original pelvis position is stored in `translation`.

### fit3d
To run IK on fit3D, execute the following command:

```bash
python -m inverse_kinematics.ik_joints_fit3d --data_path <path/to/poses> --save_path <path_to_save_ik_results> --gpus <"list of gpus"> --num_procs <number of processes> --gender <gender_of_smplx_model> --split <val or train>
```

`data_path` can either be the path to the ground truth of fit3d or the path to the results file of the uplift upsample model, like generated by the eval script. Give the list of GPU numbers as a string and the number of processes per GPU that you want to use.

### ASPset
To run IK on ASPset, you need the SMPL-X-to-ASPset joints regressor. It is included in this repository at `dataset/aspset/regressor/aspset_regressor_v5.npy`. If you change the path, you need to modify `ASPSET_REGRESSOR_PATH` in ``config.py. Then, execute the following command:

```bash
python -m inverse_kinematics.ik_joints_aspset --data_path <path/to/poses> --save_path <path_to_save_ik_results> --gpus <"list of gpus"> --num_procs <number of processes> --gender <gender_of_smplx_model> --split <split>
```

Like for fit3d, `data_path` can either be the path to the ground truth of ASPset or the path to the results file of the uplift upsample model, like generated by the eval script. Give the list of GPU numbers as a string and the number of processes per GPU that you want to use.

Since the axes of the coordinate systems of ASPset and SMPL-X are different, the ASPset joints are rotated before they are used for IK. The rotation is also present in the results. This needs to be taken into account during evaluation or other further usage.


## Evaluation and B2A

### Measuring SMPL-X models: B2A

In order to measure correctly, the mapping of vertices to bodyparts needs to be known. A file containing the necessary information is provided by the [meshcapade wiki](https://meshcapade.wiki/SMPL#model-tools). Scroll to the headline *Body-part segmentation* and download the file linked to *b) SMPL-X / SUPR part segmentation* as `smplx_vert_segmentation.json`. Put it in the folder `anthro/mesurements` or adjust the variable `VERT_SEGM_PATH` in the file `config.py` to the path where you stored the file.

Measurements are possible with the `smplx_measurer` object in `anthro.measurements.measure`. You need to call the function `smplx_measurer.measure(betas, model)`. The betas are the SMPL-X beta parameters and the model is the SMPL-X model (a created object). The function returns a dictionary with the measurements. The keys are the body parts and the values are the measurements in meters. A list of the measured anthropometric values is given in the supplementary material of the paper.

### Evaluation

The evaluation code is provided in the subdirectory `evaluation`. A special format is needed for all evaluations. The results need to be given in a dictionary, the mapping is different for ASPset and fit3d. See the following sections.

#### ASPset

ASPset results need to be given in a dictionary with the following structure:
```
{
    "subject_name": {
        "clip_name": {
            "camera_name": {
                frame_num: {
                    "betas": [list of beta parameters],
                    "body_pose": [list of body pose parameters],
                    "global_orient": [global orientation,
                    "translation": [translation],
                }
            ...}
        ...}
    ...}
...}
    
```
In order to create this structure from IK results (no matter if applied to ground truth or uplift upsample results), you can run the script 
```bash
python -m evaluation.aspset_prepare_ik --input <path_to_ik_results> --output <path_to_save>
```
The script will generate files: `<save_path>_smplx_vals.pkl` and `<save_path>_res_betas.pkl`. The first contains the structure as mentioned, the second the beta parameters per subject and is used to create median and A2B beta parameters.

You can now run the evaluation on the raw results with the following command:
```bash
python -m evaluation.aspset_eval --res <smplx_vals_file> --out <save_path_for_3d_joints> --aspset_path <aspset_root_dir> --rotate_mesh
```
The `rotate_mesh` flag is optional. If set, the mesh is rotated back to the original coordinate system, which is necessary for IK executed on ASPset poses. This script creates a file and saves the 3D joint coordinates. If called a second time, it reuses the stored 3D joint coordinates and does not calculate them again. If you want to force a new calculation, add the flag `--recalc`.

### fit3D

### Evaluation with fixed beta parameters

As described in the paper, varying body shapes throughout a subject are not realistic. Therefore, we evaluate the performance of the models with fixed beta parameters. We try with the beta parameters set to the median of the beta parameters for each subject and with the A2B results, whereby the median anthropometric measurements are used as an input. In order to prepare these different sets of beta parameters, a script is provided that creates the meshes from the beta parameters and calculates the anthropometric measurements. The script can be called with the following command:
```bash
python -m evaluation.measure_betas --input <path_to_beta_params> --output <path_to_save_measurements>
```
The input should be a `<save_path>_res_betas.pkl` file created by the `aspset_prepare_ik` or `fit3d_prepare_ik` script. The output is a file containing the measurements for each subject and the median beta parameters. 

After creating the measurements, the A2B results can be obtained with the A2B inference script, as described in the *A2B Models* section:
```bash
python -m anthro.inference --measurements <path_to_measurements_file> --save_path <path_to_save_beta_predictions.pkl> --name <measurement_name>
```
You should create a pickle-file as output in order to use the beta parameters for the evaluation.

Now you can pass such files to the evaluation scripts and evaluate with them. For ASPset:
```bash
python -m evaluation.aspset_eval --res <smplx_vals_file> --out <save_path_for_3d_joints> --betas <path_to_a2b_inference_result> --aspset_path <aspset_root_dir> --rotate_mesh
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
All other functionality (Weights, B2A, Evaluation, etc.) will be added after acceptance of this paper.