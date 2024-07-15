# -*- coding: utf-8 -*-
# Copyright (c) 2022-present, Machine Learning and Computer Vision Lab, University of Augsburg
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
# author:  goldbricklemon

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import pickle
import random
import sys

import numpy as np
import time
import datetime

import torch
from torch.utils.tensorboard import SummaryWriter

from dataset import action_wise_eval
from dataset.amass.amass_sequence_generator import create_amass_h36m_datasets, create_amass_aspset_datasets
from dataset.aspset.aspset_sequence_generator import create_aspset_datasets
from dataset.fit3d.fit3d_sequence_generator import create_fit3d_datasets
from dataset.fit3d.keypoint_order_fit3d import Fit3DOrder
from uplift_upsample.experiments.load_weights import convert_tf_to_torch_weights
from uplift_upsample.model.uplift_upsample_transformer_config import UpliftUpsampleConfig

from dataset.h36m.h36m_sequence_generator import create_h36m_datasets
from dataset.h36m.keypoint_order import H36MOrder17P
from uplift_upsample.utils import path_utils, time_formatting
from uplift_upsample.utils.metric_history import MetricHistory

torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False

from uplift_upsample import eval
from uplift_upsample.model.uplift_upsample_transformer import build_uplift_upsample_transformer


def log(*args):
    print(*args)
    sys.stdout.flush()


if __name__ == '__main__':
    import argparse

    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description='2D-to-3D uplifting training for strided poseformer.')
    parser.add_argument('--config', required=False,
                        default=None,
                        metavar="/path/to/config.json",
                        help="Path to the config file. Overwrites the default configs in the code.")
    parser.add_argument('--gpu_id', required=False,
                        default=None,
                        metavar="gpu_id",
                        help='Overwrites the GPU_ID from the config',
                        type=str)
    parser.add_argument('--dataset', required=False,
                        default="h36m",
                        metavar="{h36m, amass, aspset}",
                        help='Dataset used for training')
    parser.add_argument('--dataset_val', required=False,
                        default=None,
                        metavar="{h36m, amass}",
                        help='Dataset used for validation')
    parser.add_argument('--dataset_3d_path', required=False,
                        default="./data/data_3d_h36m.npz",
                        metavar="/path/to/h36m/",
                        help='Directory of the H36m dataset')
    parser.add_argument('--amass_path', required=False,
                        default=None,
                        metavar="/path/to/amass/",
                        help='Directory of the AMASS dataset')
    parser.add_argument('--amass_frame_rate', required=False,
                        default="50",
                        metavar="<r>",
                        help='Target frame rate for amass training')
    parser.add_argument('--dataset_2d_path', required=False,
                        default="./data/data_2d_h36m_cpn_ft_h36m_dbb.npz",
                        metavar="/path/to/2d poses/",
                        help='2D pose dataset')
    parser.add_argument('--train_subset', required=False,
                        default="train",
                        metavar="<name of train subset>",
                        help="Name of the dataset subset to train on")
    parser.add_argument('--val_subset', required=False,
                        default="val",
                        metavar="<name of val subset>",
                        help="Name of the dataset subset to validate on\
                                  pass an empty string or \"none\" to disable validation.")
    parser.add_argument('--test_subset', required=False,
                        default=None,
                        metavar="<name of test subset>",
                        help="Name of the dataset subset to test on\
                                  pass an empty string or \"none\" to disable test evaluation.")
    parser.add_argument('--continue_training', required=False,
                        default=None,
                        metavar="<path to checkpoint for continuing, config file is NOT loaded>",
                        help="Try to continue a previously started training, \
                                    mainly loading the weights, optimizer state and epoch number of the latest epoch. Does not "
                             "load the config, config needs to match")
    parser.add_argument('--tb_text', required=False,
                        default=None,
                        metavar="<name of run displayed in tensorboard>",
                        help="Add a text to the tensorboard run name in order to distinguish between runs more easily.")
    parser.add_argument('--out_dir', required=True,
                        metavar="/path/to/output_directory",
                        help='Logs and checkpoint directory. Also used to search for checkpoints if continue_training is set.')

    args = parser.parse_args()
    args.val_subset = None if args.val_subset in ["none", "None", "", 0] else args.val_subset
    args.test_subset = None if args.test_subset in ["none", "None", "", 0] else args.test_subset
    args.dataset = args.dataset.lower()
    args.dataset_val = args.dataset_val.lower() if args.dataset_val is not None else None
    val_dataset_name = args.dataset if args.dataset_val is None else args.dataset_val
    current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    log(current_time)
    log("Config: ", args.config)
    log("GPU ID: ", args.gpu_id)
    log("Dataset: ", args.dataset)
    log("Dataset Val: ", args.dataset_val)
    log("Dataset 3D: ", args.dataset_3d_path)
    log("Dataset AMASS: ", args.amass_path)
    log("AMASS frame rate: ", args.amass_frame_rate)
    log("Dataset 2D: ", args.dataset_2d_path)
    log("Train subset:", args.train_subset)
    log("Val subset:", args.val_subset)
    log("Test subset:", args.test_subset)
    log("Continue Training: ", args.continue_training)
    log("Output directory ", args.out_dir)

    assert args.dataset in ["h36m", "amass", "aspset", "fit3d"]
    assert args.dataset_val in [None, "h36m", "amass", "aspset", "fit3d"]
    # Make absolute paths
    if args.dataset in ["h36m"] or args.dataset_val in ["h36m"]:
        assert args.dataset_2d_path is not None
    elif "amass" in args.dataset or args.dataset_val is not None and "amass" in args.dataset_val:
        assert args.amass_path is not None
        args.amass_frame_rate = int(args.amass_frame_rate)
    elif args.dataset not in ["aspset", "fit3d"]:
        raise ValueError(f"{args.dataset} is not included in supported datasets.")

    args.dataset_3d_path = path_utils.expandpath(args.dataset_3d_path)
    if args.amass_path is not None:
        args.amass_path = path_utils.expandpath(args.amass_path)
    if args.dataset_2d_path is not None:
        args.dataset_2d_path = path_utils.expandpath(args.dataset_2d_path)
    if args.config is not None:
        args.config = path_utils.expandpath(args.config)

    args.out_dir = path_utils.expandpath(args.out_dir)
    # Create output directory
    path_utils.mkdirs(args.out_dir)

    # Configuration
    config = UpliftUpsampleConfig(config_file=args.config)
    assert config.ARCH == "UpliftUpsampleTransformer"
    if args.gpu_id is not None:
        assert args.gpu_id.isalnum()
        config.GPU_ID = int(args.gpu_id)

    if config.WEIGHTS is not None:
        config.WEIGHTS = path_utils.expandpath(config.WEIGHTS)

    if val_dataset_name not in ["h36m"] and config.BEST_CHECKPOINT_METRIC is not None:
        config.BEST_CHECKPOINT_METRIC = config.BEST_CHECKPOINT_METRIC.replace("AW-", "")

    # TODO: Set flip order in config.json, not in code
    if args.dataset != "fit3d":
        print("--------- using Human 3.6m 17 keypoints ---------")
        config.AUGM_FLIP_KEYPOINT_ORDER = H36MOrder17P.flip_lr_indices()
    else:
        print("--------- using Fit3D 36 keypoints ---------")
        config.AUGM_FLIP_KEYPOINT_ORDER = Fit3DOrder.flip_lr_order()

    # Print config
    config.display()

    # Set GPU
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(config.GPU_ID)

    torch.manual_seed(config.SEED)
    torch.cuda.manual_seed_all(config.SEED)
    np.random.seed(config.SEED)
    random.seed(config.SEED)

    val_subset_name = None if args.dataset_val is not None else args.val_subset
    if args.dataset == "h36m":
        train_dataset, train_dataloader, val_dataset, val_dataloader, val_batches = create_h36m_datasets(
            dataset_3d_path=args.dataset_3d_path,
            dataset_2d_path=args.dataset_2d_path,
            config=config,
            train_subset=args.train_subset,
            val_subset=val_subset_name)
    elif args.dataset == "aspset":
        train_dataset, train_dataloader, val_dataset, val_dataloader, val_batches = create_aspset_datasets(
                                                                                                aspset_path=args.dataset_3d_path,
                                                                                                config=config,
                                                                                                train_subset=args.train_subset,
                                                                                                val_subset=val_subset_name)
    elif args.dataset == "fit3d":
        train_dataset, train_dataloader, val_dataset, val_dataloader, val_batches = create_fit3d_datasets(
                                                                                                        fit3d_path=args.dataset_3d_path,
                                                                                                        config=config,
                                                                                                        train_subset=args.train_subset,
                                                                                                        val_subset=val_subset_name)
    elif args.dataset == "amass":
        train_dataset, train_dataloader, val_dataset, val_dataloader, val_batches = create_amass_h36m_datasets(amass_path=args.amass_path,
                                                                                                               h36m_path=args.dataset_3d_path,
                                                                                                               config=config,
                                                                                                               train_subset=args.train_subset,
                                                                                                               val_subset=val_subset_name,
                                                                                                               target_frame_rate=args.amass_frame_rate)

    else:
        raise NotImplementedError

    if args.dataset_val is not None:
        if args.dataset_val == "h36m":
            _, _, val_dataset, val_dataloader, val_batches = create_h36m_datasets(dataset_3d_path=args.dataset_3d_path,
                                                                                  dataset_2d_path=args.dataset_2d_path,
                                                                                  config=config,
                                                                                  train_subset=None,
                                                                                  val_subset=args.val_subset)
        else:
            raise NotImplementedError

    print("val batches", val_batches)
    # Build model, optimizer, checkpoint
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = build_uplift_upsample_transformer(config=config)
    if config.WEIGHTS is not None:
        log(f"Loading weights from {config.WEIGHTS}")
        if config.WEIGHTS.endswith(".pkl"):
            with open(config.WEIGHTS, "rb") as f:
                weights = pickle.load(f)
                weights = convert_tf_to_torch_weights(weights)
        else:
            import sys
            from uplift_upsample.model import uplift_upsample_transformer, vision_transformer
            sys.modules['model.uplift_upsample_transformer'] = uplift_upsample_transformer
            sys.modules['model.vision_transformer'] = vision_transformer
            weights = torch.load(config.WEIGHTS, map_location=device)
            if hasattr(config, "WEIGHTS_NAME"):
                weights = weights[config.WEIGHTS_NAME].state_dict()
        for k in ["spatial_pos_encoding.pe", "spatial_to_temporal_mapping.weight", "head1.0.weight", "head1.0.bias", "head2.0.weight", "head2.0.bias"]:
            if weights[k].shape != model.state_dict()[k].shape: # the case when num keypoints is different
                print(f"shape mismatch for {k}, skipping load")
                del weights[k]
        model.load_state_dict(weights, strict=False)
    model = model.to(device)

    # Keep an exponential moving average of the actual model
    if config.EMA_ENABLED is True:
        log("Cloning EMA model.")
        ema_model = build_uplift_upsample_transformer(config=config)
        # Copy weights
        ema_model.load_state_dict(model.state_dict())
        ema_model = ema_model.to(device)
    else:
        ema_model = None

    log(f"Using {config.OPTIMIZER} optimizer")
    if config.OPTIMIZER == "AdamW":
        log(config.SCHEDULE_PARAMS)
        optimizer = torch.optim.AdamW(params=model.parameters(),
                                      weight_decay=config.WEIGHT_DECAY,
                                      lr=config.SCHEDULE_PARAMS["initial_learning_rate"],
                                      eps=1e-8)


        def weight_decay_hook(optimizer, args, kwargs):
            for group in optimizer.param_groups:
                change = iteration == config.STEPS_PER_EPOCH - 1  # epoch finished
                if change:
                    group['weight_decay'] = group['weight_decay'] * config.SCHEDULE_PARAMS["decay_rate"]

        if config.SCHEDULE_PARAMS["weight_decay"]:
            optimizer.register_step_post_hook(weight_decay_hook)
    elif config.OPTIMIZER == "Adam":
        optimizer = torch.optim.Adam(params=model.parameters(), lr=config.SCHEDULE_PARAMS["initial_learning_rate"])
    else:
        raise ValueError(config.OPTIMIZER)

    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=optimizer,
                                                       gamma=config.SCHEDULE_PARAMS["decay_rate"],
                                                       last_epoch=-1)

    ckp_dict = {"optimizer": optimizer,
                "model":     model,
                "epoch":     0,}
    if config.EMA_ENABLED is True:
        ckp_dict["ema_model"] = ema_model

    output_dir = os.path.join(args.out_dir, f"run_{current_time}{'_' + args.tb_text if args.tb_text is not None else ''}")
    path_utils.mkdirs(output_dir)
    checkpoint_template = os.path.join(output_dir, "cp_{:04d}.ckpt")

    # Dump complete config to json file (for archiving)
    if args.config:
        split = os.path.split(args.config)
        split_ext = os.path.splitext(split[1])
        out_path = os.path.join(output_dir, split_ext[0] + "_complete.json")
    else:
        out_path = os.path.join(output_dir, "config_complete.json")

    config.dump(config_file=out_path)

    initial_epoch = 1
    if args.continue_training is not None:
        log(f"Restoring checkpoint from {args.continue_training}")
        checkpoint = torch.load(args.continue_training, map_location=device)
        initial_epoch = checkpoint["epoch"]
        optimizer.load_state_dict(checkpoint["optimizer"].state_dict())
        model.load_state_dict(checkpoint["model"].state_dict())
        if config.EMA_ENABLED is True:
            ema_model.load_state_dict(checkpoint["ema_model"].state_dict())

        if config.SCHEDULE_PARAMS["weight_decay"]:
            for group in optimizer.param_groups:
                group['weight_decay'] = group['weight_decay'] * config.SCHEDULE_PARAMS["decay_rate"] ** initial_epoch

    global_step = (initial_epoch - 1) * config.STEPS_PER_EPOCH

    # Metrics and Tensorboard

    tb_writer = SummaryWriter(output_dir)

    prev_best_weights_path = None
    prev_best_weights_path_ema = None
    last_weights_path = None

    metric_hist = MetricHistory()
    metrics = ["loss", "MPJPE", "NMPJPE", "PAMPJPE"]
    higher_is_better = [False,  False, False, False]
    if val_dataset_name == "h36m":
        metrics += ["AW-MPJPE", "AW-NMPJPE", "AW-PAMPJPE"]
        higher_is_better += [False, False, False]
    if config.EMA_ENABLED is True:
        metrics += ["loss/ema", "MPJPE/ema", "NMPJPE/ema", "PAMPJPE/ema"]
        higher_is_better += [False, False, False, False]
        if val_dataset_name == "h36m":
            metrics += ["AW-MPJPE/ema", "AW-NMPJPE/ema", "AW-PAMPJPE/ema"]
            higher_is_better += [False, False, False]
    for m, h in zip(metrics,
                    higher_is_better):
        metric_hist.add_metric(m, higher_is_better=h)

    if config.BEST_CHECKPOINT_METRIC is not None:
        assert config.BEST_CHECKPOINT_METRIC in metrics

    tb_writer.add_scalar('train/LR', scheduler.get_last_lr()[0], 0)
    tb_writer.add_scalar('train/WD', optimizer.param_groups[0]["weight_decay"], 0)

    def train_step(keypoints2d, keypoints3d, stride_masks, cams, ema_decay, within_batch_augment=False):
        keypoints3d = keypoints3d.to(device)
        absolute_keypoints3d = keypoints3d

        if within_batch_augment:
            if len(keypoints3d.shape) == 5:  # in batch augmentation already performed
                flipped_3d = keypoints3d[:, 1]
                keypoints3d = keypoints3d[:, 0]
                flipped_2d = keypoints2d[:, 1]
                keypoints2d = keypoints2d[:, 0]
                flipped_cam = cams.clone()
            else:
                flipped_3d, flipped_2d, flipped_cam = val_dataset.flip_batch(keypoints3d, keypoints2d, cams)
            keypoints3d = torch.cat([keypoints3d, flipped_3d], dim=0)
            keypoints2d = torch.cat([keypoints2d, flipped_2d], dim=0)
            stride_masks = torch.cat([stride_masks, stride_masks], dim=0)
            if isinstance(cams, dict):
                cams["flipped_intrinsic"] = flipped_cam["intrinsic"]
                cams["flipped_extrinsic"] = flipped_cam["extrinsic"]
            else:
                cams = torch.cat([cams, flipped_cam], dim=0)

        keypoints3d = keypoints3d - keypoints3d[:, :, config.ROOT_KEYTPOINT: config.ROOT_KEYTPOINT + 1, :]
        mid_index = config.SEQUENCE_LENGTH // 2
        central_keypoints_3d = keypoints3d[:, mid_index]

        model_input = keypoints2d.to(device).float()
        if model.has_strided_input:
            masked_keypoints2d = keypoints2d * stride_masks[:, :, None, None]
            model_input = [masked_keypoints2d.to(device).float(), stride_masks.to(device)]

        optimizer.zero_grad()

        pred_keypoints_3d, pred_keypoints_3d_central = model(model_input)
        # central_loss is: (B, K)
        central_loss = torch.linalg.norm(pred_keypoints_3d_central - central_keypoints_3d, dim=-1)
        # Aggregate loss over keypoints and batch
        central_loss_agg = torch.sum(central_loss) / (config.BATCH_SIZE * config.NUM_KEYPOINTS)

        if config.TEMPORAL_TRANSFORMER_BLOCKS > 0:
            # sequence_loss is: (B, N, K)
            sequence_loss = torch.linalg.norm(pred_keypoints_3d - keypoints3d, dim=-1)
            # Aggregate loss over keypoints, sequence and batch
            sequence_loss_agg = torch.sum(sequence_loss) / (
                    config.BATCH_SIZE * config.SEQUENCE_LENGTH * config.NUM_KEYPOINTS)

            loss = (config.LOSS_WEIGHT_CENTER * central_loss_agg) + (config.LOSS_WEIGHT_SEQUENCE * sequence_loss_agg)
        else:
            # Fallback without temporal transformer blocks: disable sequence loss
            loss = (config.LOSS_WEIGHT_CENTER + config.LOSS_WEIGHT_SEQUENCE) * central_loss_agg

        loss.backward()
        optimizer.step()

        if config.EMA_ENABLED is True:
            with torch.no_grad():
                for param_s, param_t in zip(model.parameters(), ema_model.parameters()):
                    param_t.data = param_t.data * ema_decay + param_s.data * (1. - ema_decay)
                for buffer_s, buffer_t in zip(model.buffers(), ema_model.buffers()):
                    buffer_t.data = buffer_t.data * ema_decay + buffer_s.data * (1. - ema_decay)

        return loss

    def validation(val_model, previous_best_weights_path, ep, prefix=""):
        model.eval()
        if (ep % config.VALIDATION_INTERVAL == 0 or ep == 0 and config.EVAL_BEFORE_TRAIN_START) and args.val_subset is not None:
            log(f"Running validation on {config.VALIDATION_EXAMPLES} examples")
            print_pref = "" if len(prefix) == 0 else prefix + " "
            val_start = time.time()
            val_gt_keypoints3d = list()
            val_pred_keypoints3d = list()
            val_gt_subjects = list()
            val_gt_actions = list()
            examples = 0
            val_loss = 0
            for b_i, (
                    val_sequences_3d, val_sequences_2d, val_sequences_mask,
                    val_sequence_camera_params, val_sequence_subjects, val_sequence_actions, _,
                    val_stride_masks) in enumerate(
                    val_dataloader):

                pred_keypoints3d, v_loss = val_step(val_model, keypoints2d=val_sequences_2d, keypoints3d=val_sequences_3d,
                                                    stride_masks=val_stride_masks)
                val_loss += v_loss.item()

                if config.EVAL_FLIP is True:
                    flipped_sequences_2d = val_sequences_2d
                    flipped_sequences_2d = torch.concat([flipped_sequences_2d[:, :, :, :1] * -1.,
                                                         flipped_sequences_2d[:, :, :, 1:]], dim=-1)
                    flipped_sequences_2d = flipped_sequences_2d[:, :, config.AUGM_FLIP_KEYPOINT_ORDER]

                    flipped_sequences_3d = val_sequences_3d
                    flipped_sequences_3d = torch.concat([flipped_sequences_3d[:, :, :, :1] * -1.,
                                                         flipped_sequences_3d[:, :, :, 1:]], dim=-1)
                    flipped_sequences_3d = flipped_sequences_3d[:, :, config.AUGM_FLIP_KEYPOINT_ORDER]

                    flipped_pred_keypoints_3d, _ = val_step(val_model, keypoints2d=flipped_sequences_2d,
                                                            keypoints3d=flipped_sequences_3d,
                                                            stride_masks=val_stride_masks)
                    flipped_pred_keypoints_3d = torch.concat([flipped_pred_keypoints_3d[:, :, :1] * -1.,
                                                              flipped_pred_keypoints_3d[:, :, 1:]], dim=-1)
                    flipped_pred_keypoints_3d = flipped_pred_keypoints_3d[:, config.AUGM_FLIP_KEYPOINT_ORDER]

                    pred_keypoints3d += flipped_pred_keypoints_3d
                    pred_keypoints3d /= 2.

                # Only collect as many examples as needed
                examples_to_include = min(config.BATCH_SIZE, config.VALIDATION_EXAMPLES - examples)
                # Perform root-shift right before metric calculation
                val_sequences_3d = val_sequences_3d - val_sequences_3d[:, :,
                                                      config.ROOT_KEYTPOINT: config.ROOT_KEYTPOINT + 1, :]
                val_central_keypoints_3d = val_sequences_3d[:, mid_index]
                val_gt_keypoints3d.extend(val_central_keypoints_3d[:examples_to_include].numpy())
                val_pred_keypoints3d.extend(pred_keypoints3d[:examples_to_include].cpu().numpy())
                val_gt_subjects.extend(val_sequence_subjects[:examples_to_include].numpy())
                val_gt_actions.extend(val_sequence_actions[:examples_to_include].numpy())
                examples += examples_to_include

            val_gt_keypoints3d = np.stack(val_gt_keypoints3d, axis=0).astype(np.float64)
            # Add dummy valid flag
            val_gt_keypoints3d = np.concatenate([val_gt_keypoints3d, np.ones(val_gt_keypoints3d.shape[:-1] + (1,))],
                                                axis=-1)
            val_pred_keypoints3d = np.stack(val_pred_keypoints3d, axis=0).astype(np.float64)
            val_gt_subjects = np.stack(val_gt_subjects, axis=0)
            val_gt_actions = np.stack(val_gt_actions, axis=0)
            assert b_i == (val_batches - 1)

            if val_dataset_name == "h36m":
                # Run H36m 3D evaluation
                frame_results, action_wise_results, _ = action_wise_eval.h36_action_wise_eval(
                        pred_3d=val_pred_keypoints3d,
                        gt_3d=val_gt_keypoints3d,
                        actions=val_gt_actions,
                        root_index=config.ROOT_KEYTPOINT)
            else:
                frame_results = action_wise_eval.frame_wise_eval(
                        pred_3d=val_pred_keypoints3d,
                        gt_3d=val_gt_keypoints3d,
                        root_index=config.ROOT_KEYTPOINT)

            val_duration = time.time() - val_start
            val_duration_string = time_formatting.format_time(val_duration)

            log(
                    f"{print_pref}Finished validation in {val_duration_string}, loss: {val_loss / b_i:.6f}, "
                    f"{print_pref}MPJPE: {frame_results[0]['mpjpe']:.2f}, "
                    f"{print_pref}NMPJPE: {frame_results[0]['nmpjpe']:.2f}, "
                    f"{print_pref}PAMPJPE: {frame_results[0]['pampjpe']:.2f}, "
                    )
            if val_dataset_name == "h36m":
                log(
                        f"{print_pref}AW-MPJPE: {action_wise_results['mpjpe']:.2f}, "
                        f"{print_pref}AW-NMPJPE: {action_wise_results['nmpjpe']:.2f}, "
                        f"{print_pref}AW-PAMPJPE: {action_wise_results['pampjpe']:.2f}, "
                        )

            tb_pref = f"{prefix}/" if len(prefix) > 0 else ""
            tb_writer.add_scalar(f'{tb_pref}val/loss', val_loss / b_i, ep)
            tb_writer.add_scalar(f'{tb_pref}val/MPJPE', frame_results[0]['mpjpe'], ep)
            tb_writer.add_scalar(f'{tb_pref}val/NMPJPE', frame_results[0]['nmpjpe'], ep)
            tb_writer.add_scalar(f'{tb_pref}val/PAMPJPE', frame_results[0]['pampjpe'], ep)
            if val_dataset_name == "h36m":
                tb_writer.add_scalar(f'{tb_pref}val/AW-MPJPE', action_wise_results['mpjpe'], ep)
                tb_writer.add_scalar(f'{tb_pref}val/AW-NMPJPE', action_wise_results['nmpjpe'], ep)
                tb_writer.add_scalar(f'{tb_pref}val/AW-PAMPJPE', action_wise_results['pampjpe'], ep)

            tb_pref = f"/{prefix}" if len(prefix) > 0 else ""
            metric_hist.add_data(f"loss{tb_pref}", val_loss / b_i, ep)
            metric_hist.add_data(f"MPJPE{tb_pref}", frame_results[0]['mpjpe'], ep)
            metric_hist.add_data(f"NMPJPE{tb_pref}", frame_results[0]['nmpjpe'], ep)
            metric_hist.add_data(f"PAMPJPE{tb_pref}", frame_results[0]['pampjpe'], ep)
            if val_dataset_name == "h36m":
                metric_hist.add_data(f"AW-MPJPE{tb_pref}", action_wise_results['mpjpe'], ep)
                metric_hist.add_data(f"AW-NMPJPE{tb_pref}", action_wise_results['nmpjpe'], ep)
                metric_hist.add_data(f"AW-PAMPJPE{tb_pref}", action_wise_results['pampjpe'], ep)

            if config.BEST_CHECKPOINT_METRIC is not None and args.val_subset is not None:
                # Save best checkpoint as .h5
                best_value, best_epoch = metric_hist.best_value(f"{config.BEST_CHECKPOINT_METRIC}{tb_pref}")
                if best_epoch == ep:
                    print(
                            f"Saving currently best checkpoint @ epoch {best_epoch} ({config.BEST_CHECKPOINT_METRIC}: {best_value}) as .pth:")
                    weights_path = os.path.join(output_dir, f"best_weights_{prefix + '_' if len(prefix) > 0 else ''}{best_epoch:04d}.pth")
                    torch.save(ckp_dict, weights_path)

                    if previous_best_weights_path is not None and os.path.exists(previous_best_weights_path):
                        os.remove(previous_best_weights_path)

                    previous_best_weights_path = weights_path

        return previous_best_weights_path

    def val_step(val_model, keypoints2d, keypoints3d, stride_masks):
        keypoints3d = keypoints3d.to(device)
        keypoints3d = keypoints3d - keypoints3d[:, :, config.ROOT_KEYTPOINT: config.ROOT_KEYTPOINT + 1, :]
        mid_index = config.SEQUENCE_LENGTH // 2
        central_keypoints_3d = keypoints3d[:, mid_index]

        model_input = keypoints2d.to(device).float()
        if model.has_strided_input:
            masked_keypoints2d = keypoints2d * stride_masks[:, :, None, None]
            model_input = [masked_keypoints2d.to(device).float(), stride_masks.to(device)]

        with torch.no_grad():
            pred_keypoints_3d, pred_keypoints_3d_central = val_model(model_input)
            # central_loss is: (B, K)
            central_loss = torch.linalg.norm(pred_keypoints_3d_central - central_keypoints_3d, dim=-1)
            # Aggregate loss over keypoints and batch
            central_loss = torch.sum(central_loss) / (config.BATCH_SIZE * config.NUM_KEYPOINTS)

            if config.TEMPORAL_TRANSFORMER_BLOCKS > 0:
                # sequence_loss is: (B, N, K)
                sequence_loss = torch.linalg.norm(pred_keypoints_3d - keypoints3d, dim=-1)
                # Aggregate loss over keypoints, sequence and batch
                sequence_loss = torch.sum(sequence_loss) / (
                        config.BATCH_SIZE * config.SEQUENCE_LENGTH * config.NUM_KEYPOINTS)

                loss = (config.LOSS_WEIGHT_CENTER * central_loss) + (config.LOSS_WEIGHT_SEQUENCE * sequence_loss)
            else:
                # Fallback without temporal transformer blocks: disable sequence loss
                loss = (config.LOSS_WEIGHT_CENTER + config.LOSS_WEIGHT_SEQUENCE) * central_loss

        return pred_keypoints_3d_central, loss


    train_iter = iter(train_dataloader)
    # Train loop
    ema_decay = 0
    mid_index = config.SEQUENCE_LENGTH // 2
    epoch_duration = 0.
    # Epochs use 1-based index

    if config.EVAL_BEFORE_TRAIN_START:
        prev_best_weights_path = validation(model, prev_best_weights_path, 0)
        if config.EMA_ENABLED:
            prev_best_weights_path_ema = validation(ema_model, prev_best_weights_path_ema, 0, prefix="ema")

    for epoch in range(initial_epoch, config.EPOCHS + 1):

        ckp_dict["epoch"] = epoch
        epoch_start = time.time()
        log(f"## EPOCH {epoch} / {config.EPOCHS}")
        epoch_loss = 0
        # (Global) Steps use 0-based index
        model.train()
        for iteration in range(config.STEPS_PER_EPOCH):
            tick = time.time()
            if config.EMA_ENABLED:
                ema_decay = min(config.EMA_DECAY, (1.0 + global_step) / (10.0 + global_step))

            try:
                data = next(train_iter)
            except StopIteration:
                train_iter = iter(train_dataloader)
                data = next(train_iter)

            sequences_3d, sequences_2d, sequences_mask, sequence_camera_params, _, _, _, stride_masks = data
            if isinstance(sequence_camera_params, list):
                sequence_camera_params = sequence_camera_params[0]
            loss = train_step(keypoints2d=sequences_2d, keypoints3d=sequences_3d,
                              stride_masks=stride_masks, cams=sequence_camera_params,
                              ema_decay=ema_decay, within_batch_augment=config.IN_BATCH_AUGMENT)

            epoch_loss += loss.item()
            tock = time.time()
            step_duration = tock - tick
            epoch_duration = tock - epoch_start

            if iteration % config.CHECKPOINT_INTERVAL == 0:
                eta = ((config.STEPS_PER_EPOCH - iteration - 1) / (iteration + 1)) * epoch_duration
                eta_string = time_formatting.format_time(eta)
                log(f"{iteration}/{config.STEPS_PER_EPOCH} @ Epoch {epoch} "
                    f"(Step {step_duration:.3f}s, ETA {eta_string}): "
                    f"Mean loss {epoch_loss/(iteration+1):.6f}, lr: {scheduler.get_last_lr()[0]:.6f}")

            global_step += 1
        scheduler.step()

        # Checkpoint
        if epoch % config.CHECKPOINT_INTERVAL == 0:
            torch.save(ckp_dict, checkpoint_template.format(epoch))

        if config.STEPS_PER_EPOCH > 0:
            epoch_duration_string = time_formatting.format_time(epoch_duration)
            mean_step_duration_string = epoch_duration / config.STEPS_PER_EPOCH
            log(f"Finished epoch {epoch} in {epoch_duration_string}, {mean_step_duration_string:.3f}s/step")
            tb_writer.add_scalar('train/loss', epoch_loss/config.STEPS_PER_EPOCH, epoch)
            tb_writer.add_scalar('train/LR', scheduler.get_last_lr()[0], epoch)
            tb_writer.add_scalar('train/WD', optimizer.param_groups[0]["weight_decay"], epoch)
            tb_writer.add_scalar('train/step_duration', epoch_duration / config.STEPS_PER_EPOCH, epoch)

            prev_best_weights_path = validation(model, prev_best_weights_path, epoch)
            if config.EMA_ENABLED:
                prev_best_weights_path_ema = validation(ema_model, prev_best_weights_path_ema, epoch, prefix="ema")

        last_weights_path = os.path.join(output_dir, f"last_weights.pth")
        torch.save(ckp_dict, last_weights_path)

    tb_writer.close()

    if args.val_subset is not None:
        log(f"Best checkpoint results:")
        if config.BEST_CHECKPOINT_METRIC is not None:
            metric_hist.print_all_for_best_metric(metric=config.BEST_CHECKPOINT_METRIC)
        else:
            metric_hist.print_best()

    if args.test_subset is not None and val_dataset_name in ["h36m"]:
        if config.BEST_CHECKPOINT_METRIC is not None and args.val_subset is not None:
            print("Eval best weights")
            eval_weights_path = prev_best_weights_path
        else:
            print("Eval last weights")
            eval_weights_path = last_weights_path

        eval.run_eval_multi_mask_stride(config=config,
                                        dataset_name=val_dataset_name,
                                        dataset_path=args.dataset_3d_path,
                                        dataset2d_path=args.dataset_2d_path,
                                        test_subset=args.test_subset,
                                        weights_path=eval_weights_path,
                                        model=None,
                                        action_wise=True)
    elif args.test_subset is not None:
        if config.BEST_CHECKPOINT_METRIC is not None and args.val_subset is not None:
            print("Eval best weights")
            eval_weights_path = prev_best_weights_path
        else:
            print("Eval last weights")
            eval_weights_path = last_weights_path

        eval.run_eval_multi_mask_stride(config=config,
                                        dataset_name=val_dataset_name,
                                        dataset_path=args.dataset_3d_path,
                                        dataset2d_path=args.dataset_2d_path,
                                        test_subset=args.test_subset,
                                        weights_path=eval_weights_path,
                                        model=None,
                                        action_wise=False)

    log("Done.")
