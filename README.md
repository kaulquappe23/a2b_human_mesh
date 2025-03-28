# Leveraging Anthropometric Measurements to Improve Human Mesh Estimation and Ensure Consistent Body Shapes

This repository contains all functionality from the paper "[Leveraging Anthropometric Measurements to Improve Human Mesh Estimation and Ensure Consistent Body Shapes](https://arxiv.org/abs/2409.17671)". 

The paper is accepted for [CVsports'25](https://vap.aau.dk/cvsports/) at [CVPR 2025](https://cvpr.thecvf.com/Conferences/2025). 

The trainings and evaluations mentioned in the paper can be reproduced with this repository. If you are only interested in the A2B models, we provide a second, reduced repository [here](https://github.com/kaulquappe23/a2b).

![Example](/examples/visualization.png)
More qualitative examples can be found in the folder `examples`.

## Installation

Create a new anaconda environment by running 
```bash
conda env create -f environment.yml
```
and activate the environment with
```bash
conda activate a2b_hme
```
For evaluation and fit3D, the SMPL-X models are needed. If you just want to test the A2B models, the SMPL-X models are not necessary. If you need them, download the SMPL-X models from the [official SMPL-X repository](https://smpl-x.is.tue.mpg.de) and place them in the folder `smpl-models/smplx`. Alternatively, you can modify the variable `SMPL_MODEL_DIR` in the file `config.py` and set it to the directory where you have stored the model files.

## Reproducing the Results

In order to reproduce the results, you need to follow the steps in the following order:
1. Generate 2D predictions. We finetuned ViTPose on the ASPset and fit3D datasets. We provide the 2D predictions for ASPset in the directory `dataset/aspset/vitpose` and for fit3D, we exemplary provide the validation results for one cross-validation run with s11 as the validation subject [here](https://mediastore.rz.uni-augsburg.de/get/0fVLo70EAJ/). If you want to generate the 2D predictions yourself, you can use the code from the [official ViTPose repository](https://github.com/ViTAE-Transformer/ViTPose). Note that we finetuned ViTPose for our final results.
2. Estimate 3D poses with Uplift and Upsample. We provide the code for training and evaluation in the subdirectory `uplift_upsample`. The training and evaluation steps are described in Section [Uplift Upsample Evaluation](#uplift-upsample-evaluation). Be careful to choose the settings correctly for the specific dataset since they are a little different for both datasets, which is described in detail in the mentioned section. 
3. Run IK on the UU results. Warning: This might take a lot of time, especially for fit3D since it is a large dataset. The steps are described in Section [Inverse Kinematics](#inverse-kinematics). The command line parameter `split` should be set to `val` for fit3D and `test` for ASPset. We used a neutral gender in our experiments. Note that you need to add `_stride_5` to the path with the stored UU results. UU evaluates on different strides, we choose the lowest stride 5 since it is the best.
4. Create A2B body shapes. At first, you need to re-organize the IK results, which is explained in Section [Evaluation](#evaluation): you need to run a `<dataset>_prepare_ik` script which will generate two files containing all SMPL-X parameters and a file with the beta parameters only. This file can now be used to create anthropometric measurements (see Section [Evaluation with fixed beta parameters](#evaluation-with-fixed-beta-parameters). The measurements can then be used as an input to the A2B models. For ASPset, we provide the anthropometric measurements that we derived from IK executed on the ground truth poses in the folder `dataset/aspset/measurements`. For fit3D, you can download the ground truth measurements [here](https://mediastore.rz.uni-augsburg.de/get/dIYApu4Mey/). You can directly pass this file to the A2B models as described in Section [A2B Models](#a2b-models). 
5. Evaluate the results. The evaluation procedure is described in Section [Evaluation](#evaluation), for evaluation with consistent body shapes, see Section [Evaluation with fixed beta parameters](#evaluation-with-fixed-beta-parameters). Pass the results of the A2B models as the `betas` parameter to use them during evaluation.


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

It is further possible to pass a pickle-file that contains a dictionary mapping from subject names to a dictionary of anthropometric measurements, whereby the measurements should be a list. This functionality is used for the evaluation process (see Section *Evaluation*). The list of measurements is reduced with a median operation and then passed to the A2B models. The output is then a single set of beta parameters per subject per A2B model. The dictionary might also contain the median of the originally estimated beta parameters with the key `original_median_betas`. This entry will be ignored and just passed to the final result. The final result is identical to the example output above, except that it contains the original median beta parameters, too.

## Inverse Kinematics

We provide the code to run inverse kinematics (IK) on the ASPset as well as fit3D ground truth data and results from the Uplift Upsample model. Running it on any other set of 3D poses can be added analogously. The code is provided in the subdirectory `inverse_kinematics`. You need vposer to run the code. Download the pretrained vposer model from the [official VPoser repository](https://smpl-x.is.tue.mpg.de) and place it in the folder `inverse_kinematics/V02_05`. Alternatively, you can modify the variable `VPOSER_DIR` in the file `config.py` and set it to the directory where you have stored the model files. 

The results are stored per video. A npz-file is generated per video, containing the results as a dictionary with the SMPL-X parameters (`trans`, `root_orient`, `pose_body`, `betas`) and the values per video frame. Before executing IK, the poses are converted such that the pelvis is in the origin. This speeds up the process. In order to convert the poses back to the original coordinate system, the original pelvis position is stored in `translation`.

### fit3d
To run IK on fit3D, execute the following command:

```bash
python -m inverse_kinematics.ik_joints_fit3d --data_path <path/to/poses> --save_path <path_to_save_ik_results> --gpus <"list of gpus"> --num_procs <number of processes> --gender <gender_of_smplx_model> --split <val or train> --subject_val s<subj_num>
```

`data_path` can either be the path to the ground truth of fit3d or the path to the results file of the uplift upsample model, like generated by the eval script. Give the list of GPU numbers as a string and the number of processes per GPU that you want to use. If you want to execute the IK with the ground truth, you need to specify which subject is the validation set and if you want to calculate the results for the validation or training set. 

### ASPset
To run IK on ASPset, you need the SMPL-X-to-ASPset joints regressor. It is included in this repository at `dataset/aspset/regressor/aspset_regressor_v5.npy`. If you change the path, you need to modify `ASPSET_REGRESSOR_PATH` in ``config.py. Then, execute the following command:

```bash
python -m inverse_kinematics.ik_joints_aspset --data_path <path/to/poses> --save_path <path_to_save_ik_results> --gpus <"list of gpus"> --num_procs <number of processes> --gender <gender_of_smplx_model> --split <split>
```

Like for fit3d, `data_path` can either be the path to the ground truth of ASPset or the path to the results file of the uplift upsample model, like generated by the eval script. Give the list of GPU numbers as a string and the number of processes per GPU that you want to use.

Since the axes of the coordinate systems of ASPset and SMPL-X are different, the ASPset joints are rotated before they are used for IK. The rotation is also present in the results. This needs to be taken into account during evaluation or other further usage.


## Measuring SMPL-X models: B2A

In order to measure correctly, the mapping of vertices to bodyparts needs to be known. A file containing the necessary information is provided by the [meshcapade wiki](https://meshcapade.wiki/SMPL#model-tools). Scroll to the headline *Body-part segmentation* and download the file linked to *b) SMPL-X / SUPR part segmentation* as `smplx_vert_segmentation.json`. Put it in the folder `anthro/mesurements` or adjust the variable `VERT_SEGM_PATH` in the file `config.py` to the path where you stored the file.

Measurements are possible with the `smplx_measurer` object in `anthro.measurements.measure`. You need to call the function `smplx_measurer.measure(betas, model)`. The betas are the SMPL-X beta parameters and the model is the SMPL-X model (a created object with the correct gender). The function returns a dictionary with the measurements. The keys are the body parts and the values are the measurements in meters. A list of the measured anthropometric values is given in Section [A2B Models](#a2b-models).

## Evaluation

The evaluation code is provided in the subdirectory `evaluation`. A special format is needed for all evaluations. The results need to be given in a dictionary, the mapping is a little different for ASPset and fit3D. See the following sections.

### ASPset

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

fit3D results need to be given in a dictionary with the following structure:
```
{
    "subject_name": {
        "camera_name": {
            "action_name": {
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
python -m evaluation.fit3d_prepare_ik --input <path_to_ik_results> --output <path_to_save>
```
The script will generate files: `<save_path>_smplx_vals.pkl` and `<save_path>_res_betas.pkl`. The first contains the structure as mentioned, the second the beta parameters per subject and is used to create median and A2B beta parameters.

You can now run the evaluation on the raw results with the following command:
```bash
python -m evaluation.fit3d_eval --res <smplx_vals_file> --out <save_path_for_3d_joints> --fit3d_path <aspset_root_dir> 
```
This script creates a file and saves the 3D joint coordinates. If called a second time, it reuses the stored 3D joint coordinates and does not calculate them again. If you want to force a new calculation, add the flag `--recalc`.

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
In order to train Uplift Upsample on ASPset, you need to download the dataset with the code provided by the [official  ASPset repository](https://github.com/anibali/aspset-510). Training on ASPset requires 2D predictions from ViTPose as an input. These files are expected as csv files for each split. We provide the files for ASPset in the directory `dataset/aspset/vitpose`. You can also put them elsewhere and update the config file (setting `PREDICTION_PATH_2D`). We further use the pretrained weights from the AMASS pretrained and Human3.6m finetuned variant. Therefore, download `h36m_351_pt.h5` from the [official Uplift Upsample repository](https://github.com/goldbricklemon/uplift-upsample-3dhpe). Our repository provides a converter from their Tensorflow weights in `h5` format to PyTorch weights. Put the downloaded pretrained weights in the following folder: `./uplift_upsample/pretrained_weights`.

You can train uplift upsample on ASPset with the following call. Replace `dataset_3d_path` with the path where you stored the ASPset files.
```bash
python -m uplift_upsample.train --config ./uplift_upsample/experiments/aspset_351.json --out_dir ./uplift_upsample/out --dataset_3d_path <path/to/aspset> --train_subset train --val_subset val --dataset aspset --gpu_id 0
```

### Training on fit3D
Training on fit3D is executed analogous to ASPset. The csv files for one exemplary cross-validation run (s11 as the validation subject) for fit3D can be downloaded (here)[https://mediastore.rz.uni-augsburg.de/get/0fVLo70EAJ/]. Since we use cross-validation here, you need to specify in the config file, on which validation subject you want to train (adjust the parameter named `PREDICTION_PATH_2D` to the location where the `train.csv`and `val.csv` files are stored). The vitpose files should structured such that the folder name is the subject name of the validation subject.

### Training on AMASS and Human3.6m
Follow the instructions of the [original repository](https://github.com/goldbricklemon/uplift-upsample-3dhpe) to setup the dataset(s), download the pre-trained models, and run the code. Since the code is a sub-repository here, the only difference for running is that you need to add the `uplift_upsample` prefix to the train/eval calls, etc. Pretrained weights can be downloaded from the original repository in `.h5` format and reused in this repository.

### Uplift Upsample Evaluation

You can either evaluate your own trained model, or use our pretrained weights. We provide the weights for the ASPset and fit3D datasets. For fit3D, we perform a leave-one-out cross-validation. Therefore, pretrained weights are available for each left-out subject. The left-out subject is contained in the filename. The weights can be downloaded [here](https://mediastore.rz.uni-augsburg.de/get/E_hsThx4v0/). Put the weights in the folder `./uplift_upsample/pretrained_weights`. For fit3d, the model name contains the left-out subject. 

Run the evaluation with the following command:
```bash
python -m uplift_upsample.eval --dataset <aspset/fit3d> --weights <path/to/weights/file> --config ./uplift_upsample/experiments/<config>.json --savefile <path/wehere/results/are/stored.pkl> --dataset_3d_path <path/to/aspset> --test_subset <val/test> --gpu_id 0 --model <model/ema_model>
```
For fit3D, you need to use `val` as the `test_subset` and `model` as the `model` name. For ASPset, use `test` as the `test_subset` and the `ema_model` as the `model`.
The savefile can now be used to run IK on. We always use the stride 5 evaluation since it is the best, but you can also try with other strides. 

# Citation

In case this work is useful for your research, please consider citing:
```bibtex
@article{ludwig2024leveraging,
  title={Leveraging Anthropometric Measurements to Improve Human Mesh Estimation and Ensure Consistent Body Shapes},
  author={Ludwig, Katja and Lorenz, Julian and Kienzle, Daniel and Bui, Tuan and Lienhart, Rainer},
  journal={arXiv preprint arXiv:2409.17671},
  year={2024}
}
```