# -*- coding: utf-8 -*-
"""
Created on 25.10.23



"""
import datetime
import os
import pickle
import sys
import time

import numpy as np
import torch.cuda
import torch
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False

from dataset.action_wise_eval import interpolate_between_keyframes, compute_and_log_metrics
from dataset.aspset.aspset_sequence_generator import create_aspset_dataloaders
from dataset.fit3d.fit3d_sequence_generator import create_fit3d_dataloaders
from dataset.fit3d.keypoint_order_fit3d import Fit3DOrder
from dataset.h36m.h36m_sequence_generator import create_h36m_dataloaders
from uplift_upsample.model.uplift_upsample_transformer import build_uplift_upsample_transformer
from uplift_upsample.model.uplift_upsample_transformer_config import UpliftUpsampleConfig

from tqdm import tqdm

from uplift_upsample.utils import path_utils, time_formatting


def log(*args):
    print(*args)
    sys.stdout.flush()


def run_eval(config: UpliftUpsampleConfig, stride, dataset_name, dataset_path, dataset2d_path, test_subset,
             weights_path=None, model=None, action_wise=True, model_name="model", save_results=None):
    """
    Run H3.6m evaluation with the given model.
    :param config: Model config
    :param dataset: dataset name
    :param dataset_path: 3D dataset .npz
    :param dataset2d_path: 2D dataset .npz
    :param test_subset: Dataset split to evaluate on
    :param weights_path: Path to weight file to load (optional).
    :param model: Model to evaluate.
    :param action_wise: Perform action-wise evaluation (True) or frame-wise evaluation (False)
    :return:
    """

    assert not (weights_path is None and model is None)

    # Build model, optimizer, checkpoint
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if model is None:
        model = build_uplift_upsample_transformer(config=config)
        if weights_path is not None:
            log(f"Loading weights from {weights_path}")
            weights = torch.load(weights_path, map_location=device, weights_only=False)
            if model_name in weights:
                weights = weights[model_name].state_dict()
            model.load_state_dict(weights)

    elif weights_path is not None:
        log(f"Using provided model. Ignoring the given weights path: {weights_path}")

    model.eval()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)

    if dataset_name == "h36m":
        dataset, data_loader = create_h36m_dataloaders(dataset_3d_path=dataset_path,
                                                       dataset_2d_path=dataset2d_path,
                                                       split="test",
                                                       config=config,
                                                       selection=test_subset)
    elif dataset_name == "aspset":
        dataset, data_loader = create_aspset_dataloaders(dataset_3d_path=dataset_path,
                                                       split="test",
                                                       config=config,
                                                       selection=test_subset)
    elif dataset_name == "fit3d":
        dataset, data_loader = create_fit3d_dataloaders(dataset_3d_path=dataset_path,
                                                         split="val",
                                                         config=config,
                                                         selection=test_subset)
    else:
        raise RuntimeError("Invalid dataset name")

    # Test loop
    log(f"Running evaluation on '{test_subset}' with {len(dataset)} examples")
    start = time.time()
    test_gt_keypoints3d = list()
    test_pred_keypoints3d = list()
    test_gt_subjects = list()
    test_gt_actions = list()
    test_gt_indices = list()
    test_gt_cam_params = None
    test_gt_cam_names = list()
    examples = 0
    mid_index = config.SEQUENCE_LENGTH // 2

    for b_i, (
            test_sequences_3d, test_sequences_2d, test_sequences_mask, test_sequence_camera_params,
            test_sequence_subjects,
            test_sequence_actions, test_index, test_stride_masks) in enumerate(tqdm(
        data_loader)):

        model_input = test_sequences_2d.to(device).float()
        if model.has_strided_input:
            masked_keypoints2d = test_sequences_2d * test_stride_masks[:, :, None, None]
            model_input = [masked_keypoints2d.to(device).float(), test_stride_masks.to(device)]

        with torch.no_grad():
            pred_sequence_keypoints3d, pred_keypoints3d = model(model_input)

        if config.EVAL_FLIP is True:
            flipped_sequences_2d = test_sequences_2d
            flipped_sequences_2d = torch.concat([flipped_sequences_2d[:, :, :, :1] * -1.,
                                              flipped_sequences_2d[:, :, :, 1:]], dim=-1)
            flipped_sequences_2d = flipped_sequences_2d[:, :, config.AUGM_FLIP_KEYPOINT_ORDER]

            model_input = flipped_sequences_2d.to(device).float()
            if model.has_strided_input:
                masked_keypoints2d = flipped_sequences_2d * test_stride_masks[:, :, None, None]
                model_input = [masked_keypoints2d.to(device).float(), test_stride_masks.to(device)]

            with torch.no_grad():
                flipped_pred_sequence_keypoints_3d, flipped_pred_keypoints_3d = model(model_input)

            flipped_pred_keypoints_3d = torch.concat([flipped_pred_keypoints_3d[:, :, :1] * -1.,
                                                   flipped_pred_keypoints_3d[:, :, 1:]], dim=-1)
            flipped_pred_keypoints_3d = flipped_pred_keypoints_3d[:, config.AUGM_FLIP_KEYPOINT_ORDER]

            pred_keypoints3d += flipped_pred_keypoints_3d
            pred_keypoints3d /= 2.

            if model.full_output and config.TEMPORAL_TRANSFORMER_BLOCKS > 0:
                flipped_pred_sequence_keypoints_3d = torch.concat([flipped_pred_sequence_keypoints_3d[:, :, :, :1] * -1.,
                                                                flipped_pred_sequence_keypoints_3d[:, :, :, 1:]],
                                                               dim=-1)
                flipped_pred_sequence_keypoints_3d = flipped_pred_sequence_keypoints_3d[:, :, config.AUGM_FLIP_KEYPOINT_ORDER]

                pred_sequence_keypoints3d += flipped_pred_sequence_keypoints_3d
                pred_sequence_keypoints3d /= 2.

        # Only collect as many examples as needed
        examples_to_include = min(config.BATCH_SIZE, len(dataset) - examples)
        # Perform root-shift right before metric calculation
        test_sequences_3d = test_sequences_3d - test_sequences_3d[:, :,
                                                config.ROOT_KEYTPOINT: config.ROOT_KEYTPOINT + 1, :]
        test_central_keypoints_3d = test_sequences_3d[:, mid_index]
        test_gt_keypoints3d.extend(test_central_keypoints_3d[:examples_to_include].numpy())
        test_pred_keypoints3d.extend(pred_keypoints3d[:examples_to_include].detach().cpu().numpy())
        test_gt_subjects.extend(test_sequence_subjects[:examples_to_include].numpy())
        test_gt_actions.extend(test_sequence_actions[:examples_to_include].numpy())
        test_gt_indices.extend(test_index[:examples_to_include].numpy())
        if isinstance(test_sequence_camera_params, list):
            cam_names = test_sequence_camera_params[-1]
            test_gt_cam_names.extend(cam_names)
            test_sequence_camera_params = test_sequence_camera_params[0]
        if test_gt_cam_params is None:
            test_gt_cam_params = test_sequence_camera_params[:examples_to_include].numpy()
        else:
            test_gt_cam_params = np.concatenate([test_gt_cam_params, test_sequence_camera_params[:examples_to_include].numpy()], axis=0)
        examples += examples_to_include

    test_gt_keypoints3d = np.stack(test_gt_keypoints3d, axis=0).astype(np.float64)
    # Add dummy valid flag
    test_gt_keypoints3d = np.concatenate([test_gt_keypoints3d, np.ones(test_gt_keypoints3d.shape[:-1] + (1,))],
                                         axis=-1)
    test_pred_keypoints3d = np.stack(test_pred_keypoints3d, axis=0).astype(np.float64)

    test_gt_subjects = np.stack(test_gt_subjects, axis=0)
    test_gt_actions = np.stack(test_gt_actions, axis=0)
    test_gt_indices = np.stack(test_gt_indices, axis=0)
    test_batches = np.ceil(len(dataset) / config.BATCH_SIZE)
    assert b_i == (test_batches - 1)

    bkup_test_pred_keypoints3d = test_pred_keypoints3d
    test_pred_keypoints3d = np.copy(bkup_test_pred_keypoints3d)

    if config.SEQUENCE_STRIDE > 1 and config.TEST_STRIDED_EVAL is True:
        log(f"Performing strided eval: Interpolating between keyframes")
        strides = np.tile([config.SEQUENCE_STRIDE], reps=(test_gt_indices.shape[0]))
        if config.EVAL_DISABLE_LEARNED_UPSAMPLING and config.MASK_STRIDE is not None:
            strides[:] = config.MASK_STRIDE

        interp_pred_keypoints3d, _ = interpolate_between_keyframes(pred3d=test_pred_keypoints3d,
                                                                   frame_indices=test_gt_indices,
                                                                   keyframe_stride=strides)

        full_pred_keypoints3d = test_pred_keypoints3d
        test_pred_keypoints3d = interp_pred_keypoints3d
    else:
        full_pred_keypoints3d = test_pred_keypoints3d

    if save_results is not None:
        data = (test_pred_keypoints3d, test_gt_subjects, test_gt_actions, test_gt_indices, test_gt_cam_params, test_gt_keypoints3d,
                dataset.conversion_dicts, test_gt_cam_names)
        savefile = f"{save_results[:save_results.rfind('.')]}_stride_{stride}{save_results[save_results.rfind('.'):]}"
        with open(savefile, 'wb') as f:
            pickle.dump(data, f)

    log("")
    log("### Evaluation on ALL FRAMES ####")
    log("")

    compute_and_log_metrics(pred3d=test_pred_keypoints3d, gt3d=test_gt_keypoints3d,
                            actions=test_gt_actions, root_index=config.ROOT_KEYTPOINT,
                            action_wise=action_wise)

    if (config.SEQUENCE_STRIDE > 1 or (
            config.MASK_STRIDE is not None and config.MASK_STRIDE > 1)) and config.TEST_STRIDED_EVAL is True:
        log("")
        log("### Evaluation on KEYFRAMES ####")
        log("")

        input_stride = config.SEQUENCE_STRIDE if config.MASK_STRIDE is None else config.MASK_STRIDE
        input_keyframes = np.equal(np.mod(test_gt_indices, input_stride), 0)

        compute_and_log_metrics(pred3d=full_pred_keypoints3d[input_keyframes],
                                gt3d=test_gt_keypoints3d[input_keyframes],
                                actions=test_gt_actions[input_keyframes], root_index=config.ROOT_KEYTPOINT,
                                action_wise=action_wise)

    duration = time.time() - start
    duration_string = time_formatting.format_time(duration)
    log(f"Finished evaluation in {duration_string}")


def run_eval_multi_mask_stride(config: UpliftUpsampleConfig, *args, **kwargs):
    # Run evaluation for each mask stride value
    config = config.copy()
    mask_stride_values = config.MASK_STRIDE
    if type(mask_stride_values) is not list:
        mask_stride_values = [mask_stride_values]
    for msv in mask_stride_values:
        config.MASK_STRIDE = msv
        if len(mask_stride_values) > 1:
            log(f"### Running evaluation for mask stride value: {msv} ###")
        run_eval(config=config, stride= msv, *args, **kwargs)
        if len(mask_stride_values) > 1:
            log(f"### Finished evaluation for mask stride value: {msv} ###")


if __name__ == '__main__':
    import argparse

    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description='3D evaluation.')
    parser.add_argument('--dataset', required=True,
                        default="fit3d",
                        metavar="dataset name, e.g. h36m",
                        help="fit3d or aspset")
    parser.add_argument('--weights', required=True,
                        default=None,
                        metavar="/path/to/weights.pth",
                        help="Path to weights file for model weight initialization.")
    parser.add_argument('--config', required=False,
                        default=None,
                        metavar="/path/to/config.json",
                        help="Path to the config file. Overwrites the default configs in the code.")
    parser.add_argument('--savefile', required=False,
                        default=None,
                        metavar="/path/to/resultsfile.pkl",
                        help="Path to the results file.")
    parser.add_argument('--gpu_id', required=False,
                        default=None,
                        metavar="gpu_id",
                        help='Overwrites the GPU_ID from the config',
                        type=str)
    parser.add_argument('--batch_size', required=False,
                        default=None,
                        metavar="batch_size",
                        help='Overwrites the BATCH_SIZE from the config',
                        type=int)
    parser.add_argument('--dataset_3d_path', required=False,
                        default="./data/data_3d_h36m.npz",
                        metavar="/path/to/h36m/.npz",
                        help='3D pose dataset')
    parser.add_argument('--dataset_2d', required=False,
                        default="./data/data_2d_h36m_cpn_ft_h36m_dbb.npz",
                        metavar="/path/to/2d poses/.npz",
                        help='2D pose dataset')
    parser.add_argument('--test_subset', required=False,
                        default="val",
                        metavar="<name of test subset>",
                        help="Name of the dataset subset to evaluate on")
    parser.add_argument('--model_name', required=False,
                        default="model",
                        metavar="<name of model in weights dict>",
                        help="Name of the dataset subset to evaluate on")
    parser.add_argument('--action_wise', dest='action_wise', action='store_false')
    parser.add_argument('--frame_wise', dest='action_wise', action='store_true')
    parser.add_argument('--forced_mask_stride', required=False,
                        default=None,
                        metavar="forced_mask_stride",
                        help='Overwrites the MASK_STRIDE from the config',
                        type=int)
    parser.add_argument('--no_learned_upsampling', dest='disable_learned_upsampling', action='store_true')
    parser.set_defaults(disable_learned_upsampling=False)

    args = parser.parse_args()

    current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    log(current_time)
    log("Config: ", args.config)
    log("GPU ID: ", args.gpu_id)
    log("Batch size: ", args.batch_size)
    log("Dataset: ", args.dataset_3d_path)
    log("Dataset 2D: ", args.dataset_2d)
    log("Test subset:", args.test_subset)
    log("Action-wise:", args.action_wise)
    log("Weights:", args.weights)
    if args.disable_learned_upsampling:
        log("Disable learned upsampling:", args.disable_learned_upsampling)
    log("Forced mask stride:", args.forced_mask_stride)

    # Make absolute paths
    args.dataset_3d_path = path_utils.expandpath(args.dataset_3d_path)
    args.dataset_2d = path_utils.expandpath(args.dataset_2d)
    if args.config is not None:
        args.config = path_utils.expandpath(args.config)
    if args.weights is not None:
        args.weights = path_utils.expandpath(args.weights)

    # Configuration
    config = UpliftUpsampleConfig(config_file=args.config)
    assert config.ARCH == "UpliftUpsampleTransformer"
    if args.forced_mask_stride is not None:
        log(f"Setting mask stride to fixed value: {args.forced_mask_stride}")
        config.MASK_STRIDE = args.forced_mask_stride

    if args.gpu_id is not None:
        assert args.gpu_id.isalnum()
        config.GPU_ID = int(args.gpu_id)
    if args.batch_size is not None:
        config.BATCH_SIZE = int(args.batch_size)
    if args.disable_learned_upsampling:
        if config.MASK_STRIDE is not None:
            log("WARNING: Disabling learned upsampling. Will use pure bi-linear upsampling.")
            config.EVAL_DISABLE_LEARNED_UPSAMPLING = True

    if args.dataset == "fit3d":
        print("--------- using Fit3D 36 keypoints ---------")
        config.AUGM_FLIP_KEYPOINT_ORDER = Fit3DOrder.flip_lr_order()
    # Print config
    config.display()

    # Set GPU
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(config.GPU_ID)

    dataset_name = args.dataset
    # Run evaluation for each mask stride value
    run_eval_multi_mask_stride(config=config, dataset_name=dataset_name,
                               dataset_path=args.dataset_3d_path,
                               dataset2d_path=args.dataset_2d,
                               test_subset=args.test_subset,
                               weights_path=args.weights,
                               action_wise=False,
                               model_name=args.model_name,
                               save_results=args.savefile)