# -*- coding: utf-8 -*-
"""
Created on 25.10.23

@author: Katja

"""
import csv
import json
import os
from collections import defaultdict

import cv2
import math
import sys

import numpy as np
from aspset510 import Aspset510, Camera
from glupy.math import to_cartesian
from posekit.skeleton import skeleton_registry
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from dataset.aspset.keypoint_order import APSSet17POrder
from dataset.camera import normalize_screen_coordinates
from uplift_upsample.utils.io import SkipCSVCommentsIterator, create_dictionary


class ASPsetSequenceGenerator(Dataset):

    def __init__(self, dataset_path, subset, seq_len, subsample, stride=1, padding_type="zeros",
                 flip_augment=True, flip_lr_indices=None,
                 mask_stride=None, stride_mask_align_global=False, rand_shift_stride_mask=False, seed=0, verbose=True, prediction_path=None):

        # Build ASPset dataset
        dataset_name = "aspset"

        dataset_3d, poses_2d_dataset = load_dataset_and_2d_poses(dataset_path=dataset_path,
                                                                 dataset_name=dataset_name,
                                                                 split=subset,
                                                                 verbose=True,
                                                                 prediction_path=prediction_path)

        camera_params, poses_3d, poses_2d, subjects, actions, frame_rates, dicts = filter_and_subsample_dataset(
                dataset=dataset_3d,
                poses_2d=poses_2d_dataset,
                downsample=1,
                verbose=True)

        self.seq_len = seq_len
        self.subsample = subsample
        self.stride = stride  # this is the used output stride s_out
        self.target_frame_rate = 50
        if padding_type == "zeros":
            self.pad_type = "constant"
        elif padding_type == "copy":
            self.pad_type = "edge"
        else:
            raise ValueError(f"Padding type not supported: {padding_type}")
        self.flip_augment = flip_augment
        self.flip_lr_indices = flip_lr_indices
        self.abs_mask_stride = mask_stride
        if self.abs_mask_stride is not None:  # these are the absolute input strides s_in that are used during training
            if type(self.abs_mask_stride) is not list:
                self.abs_mask_stride = [self.abs_mask_stride]
            for ams in self.abs_mask_stride:
                assert ams >= self.stride
                assert ams % self.stride == 0
        self.stride_mask_align_global = stride_mask_align_global
        self.rand_shift_stride_mask = rand_shift_stride_mask
        self.stride_shift_rng = np.random.default_rng(seed=seed)
        self.mask_stride_rng = np.random.default_rng(seed=seed)
        if self.rand_shift_stride_mask is True:
            assert self.stride_mask_align_global is False
        self.subset = subset

        self.verbose = verbose

        if self.verbose:
            print("Generating sequences ...")

        if self.flip_augment is True:
            assert flip_lr_indices is not None

        self.poses_3d = poses_3d
        self.poses_2d = poses_2d
        self.camera_params = camera_params
        self.subjects = subjects
        self.actions = actions
        self.frame_rates = frame_rates
        self.conversion_dicts = dicts

        # Generate all central locations of sequences.
        self.sequence_locations = []
        for s_i, video_3d in enumerate(poses_3d):
            assert len(video_3d) == len(poses_2d[s_i])
            # positions are all possible frames of the video. Will be used as the target central position later
            positions = np.arange(start=0, stop=len(video_3d), step=self.subsample)
            sequence_number = np.tile([s_i], reps=(positions.shape[0]))
            frame_rates_tiled = np.tile([self.frame_rates[s_i]], reps=(positions.shape[0]))
            do_flip = np.zeros(shape=(positions.shape[0]), dtype=positions.dtype)
            if self.flip_augment:  # add flipped version already to the sequence locations
                sequence_number = np.concatenate([sequence_number, sequence_number], axis=0)
                frame_rates_tiled = np.concatenate([frame_rates_tiled, frame_rates_tiled], axis=0)
                positions = np.concatenate([positions, positions], axis=0)
                do_flip = np.concatenate([do_flip, 1 - do_flip], axis=0)

            self.sequence_locations.append(np.stack([sequence_number, positions, do_flip, frame_rates_tiled], axis=-1))

        self.sequence_locations = np.concatenate(self.sequence_locations, axis=0)

    def __len__(self):
        return len(self.sequence_locations)

    def __getitem__(self, item):
        s_i, i, do_flip, frame_rate = self.sequence_locations[item]

        frame_rate = int(frame_rate)
        stride = self.stride # s_out
        mult = 1
        assert frame_rate % self.target_frame_rate == 0
        if frame_rate != self.target_frame_rate: # adjust stride to match target frame rate
            mult = frame_rate // self.target_frame_rate
            stride *= mult

        if self.abs_mask_stride is None:
            abs_mask_stride = stride # train with s_in = s_out
        else:
            if len(self.abs_mask_stride) == 1:
                abs_mask_stride = self.abs_mask_stride[0]
            else: # randomly choose one of the possible input strides
                abs_mask_stride = self.abs_mask_stride[
                self.mask_stride_rng.integers(low=0, high=len(self.abs_mask_stride), endpoint=False)]
            abs_mask_stride *= mult # adjust to target frame rate

        mask_stride = abs_mask_stride // stride # relative stride between output and input sequences
        # the real sequence length is (self.seq_len - 1) * stride + 1, self.seq_len is the real number of frames that is used
        left = (self.seq_len - 1) * stride // 2 # number of frames to consider to the left
        right = (self.seq_len - 1) * stride - left # number of frames to consider to the right

        do_flip = do_flip == 1.
        video_3d, video_2d, camera = self.poses_3d[s_i], self.poses_2d[s_i], self.camera_params[s_i]
        subject, action = self.subjects[s_i], self.actions[s_i]
        video_len = video_3d.shape[0]
        begin, end = i - left, i + right + 1 # begin and end frame index for selected central frame
        pad_left, pad_right = 0, 0
        if begin < 0: # we would need more poses at the beginning, therefore we need to pad the sequence on the left side
            # print(f"{i} {video_len} {left} {right} {begin} {end}")
            pad_left = math.ceil(-begin / stride)
            last_pad = begin + ((pad_left - 1) * stride)
            begin = last_pad + stride
            # print(f"LEFT: {pad_left} {last_pad} {begin}")
            # print(f"{begin} {end} {stride}")
        if end > video_len: # we would need more poses at the end, therefore we need to pad the sequence on the right side
            # print(f"{i} {video_len} {left} {right} {begin} {end}")
            pad_right = math.ceil((end - video_len) / stride)
            first_pad = end - ((pad_right - 1) * stride)
            end = first_pad - stride
            # print(f"RIGHT: {pad_right} {first_pad} {end}")
            # print(f"{begin} {end} {stride}")

        # Base case:
        sequence_3d = video_3d[begin: end: stride]
        sequence_2d = video_2d[begin: end: stride]
        mask = np.ones(sequence_3d.shape[0], dtype=np.float32)
        # Pad if necessary
        if pad_left > 0 or pad_right > 0:
            # numpy constant padding defaults to 0 values
            sequence_3d = np.pad(sequence_3d, ((pad_left, pad_right), (0, 0), (0, 0)), mode=self.pad_type)
            sequence_2d = np.pad(sequence_2d, ((pad_left, pad_right), (0, 0), (0, 0)), mode=self.pad_type)
            mask = np.pad(mask, (pad_left, pad_right), mode="constant")

        # Generate stride mask that is centered on the central frame
        mid_index = self.seq_len // 2 # index of the central frame in sequence_2d and 3d
        sequence_indices = np.arange(0, self.seq_len) - mid_index
        sequence_indices *= stride # video frame indices relative to the central frame
        if self.stride_mask_align_global is True: # real (absolute) video frame indices!
            # Shift mask such that it is aligned on the global frame indices
            # This is required for inference mode
            sequence_indices += i

        elif self.rand_shift_stride_mask is True:
            # Shift stride mask randomly by [ceil(-mask_stride/2), floor(mask_stride/2)]
            max_shift = int(np.ceil((mask_stride - 1) / 2))
            endpoint = mask_stride % 2 != 0 # include max_shift in the range
            rand_shift = self.stride_shift_rng.integers(low=-max_shift, high=max_shift, endpoint=endpoint)
            rand_shift *= stride # sequence indices are in (relative) video frame indices, so we need to shift relative to the stride
            sequence_indices += rand_shift # shift the used frames randomly such that the central frame can also be masked

        stride_mask = np.equal(sequence_indices % abs_mask_stride, 0) # create mask for used and unused frames in the 2d sequence
        # stride mask contains False for every unused 2d pose and True for every used pose

        assert sequence_3d.shape[0] == self.seq_len
        assert sequence_2d.shape[0] == self.seq_len
        assert mask.shape[0] == self.seq_len
        assert stride_mask.shape[0] == self.seq_len

        camera_ = np.concatenate([camera["extrinsic"], camera["intrinsic"]], axis=0)
        if do_flip:
            # Width (or x coord) is 0 centered, so flipping is simply sign inversion
            sequence_3d = sequence_3d[:, self.flip_lr_indices].copy()
            sequence_3d[..., 0] *= -1
            sequence_2d = sequence_2d[:, self.flip_lr_indices].copy()
            sequence_2d[..., 0] *= -1
            camera_ = camera_.copy()
            # Flip cx (principal point)
            camera_[4, 2] *= -1
        camera = [camera_, camera["name"]]

        return sequence_3d.astype(float), sequence_2d.astype(float), mask, camera, subject, action, i, stride_mask

    def flip_batch(self, sequence_3d, sequence_2d, camera):
        # Width (or x coord) is 0 centered, so flipping is simply sign inversion
        sequence_3d = sequence_3d[:, :, self.flip_lr_indices].clone()
        sequence_3d[..., 0] *= -1
        sequence_2d = sequence_2d[:, :, self.flip_lr_indices].clone()
        sequence_2d[..., 0] *= -1
        intrinsic = camera[4:].clone()
        # Flip cx (principal point)
        intrinsic[0][2] *= -1
        camera[4:] = intrinsic

        return sequence_3d.float(), sequence_2d.float(), camera

def create_aspset_dataloaders(dataset_3d_path, config, split, selection):
    # The dataset is subsampled to every Nth frame (i.e. a sequence is extracted at every Nth frame)
    # The frame rate is not changed, however!
    if sys.gettrace() is None:
        WORKERS = 16
    else:
        WORKERS = 0

    subsample = config.DATASET_TRAIN_3D_SUBSAMPLE_STEP if split == "train" else config.DATASET_VAL_3D_SUBSAMPLE_STEP if (
            split == "val") else config.DATASET_TEST_3D_SUBSAMPLE_STEP
    shuffle = split == "train"
    stride_mask_rand_shift = config.STRIDE_MASK_RAND_SHIFT and split == "train"
    do_flip = split == "train" and config.AUGM_FLIP_PROB > 0 and not config.IN_BATCH_AUGMENT
    predictions = config.PREDICTION_PATH_2D if hasattr(config, "PREDICTION_PATH_2D") and config.PREDICTION_PATH_2D is not None else None
    dataset = ASPsetSequenceGenerator(dataset_path=dataset_3d_path,
                                      subset=selection,
                                      seq_len=config.SEQUENCE_LENGTH,
                                      subsample=subsample,
                                      stride=config.SEQUENCE_STRIDE,
                                      padding_type=config.PADDING_TYPE,
                                      flip_lr_indices=config.AUGM_FLIP_KEYPOINT_ORDER,
                                      mask_stride=config.MASK_STRIDE,
                                      stride_mask_align_global=split == "test",
                                      rand_shift_stride_mask=stride_mask_rand_shift,
                                      flip_augment=do_flip,
                                      seed=config.SEED,
                                      prediction_path=predictions)

    data_loader = DataLoader(
            dataset,
            batch_size=config.BATCH_SIZE // 2 if config.IN_BATCH_AUGMENT and split == "train" else config.BATCH_SIZE,
            shuffle=shuffle,
            num_workers=WORKERS,
            pin_memory=False,
            drop_last=shuffle
            )
    return dataset, data_loader

def create_aspset_datasets(aspset_path, config, train_subset, val_subset):
    train_dataset, val_dataset, val_batches = None, None, None

    for split, selection in zip(["train", "val"], [train_subset, val_subset]):
        if selection is not None:

            dataset, data_loader = create_aspset_dataloaders(aspset_path, config, split, selection)
            print(f"Sequences: {len(dataset)}")

            if split == "train":
                train_dataset = dataset
                train_dataloader = data_loader
            else:
                if config.VALIDATION_EXAMPLES < 0:
                    config.VALIDATION_EXAMPLES = len(dataset)
                assert config.VALIDATION_EXAMPLES <= len(dataset)

                val_batches = int(np.ceil(config.VALIDATION_EXAMPLES / config.BATCH_SIZE))
                val_dataset = dataset
                val_dataloader = data_loader

    return train_dataset, train_dataloader, val_dataset, val_dataloader, val_batches


def read_aspset_csv_annotations(csv_path, num_joints=APSSet17POrder.num_points):
    """
    Reads annotations from the given csv file. Additional info to be expected is dataset;image_path;video_description;frame_num
    The first line has to be a comment with the number of annotations, eg. "#123"

    Parameters:
        - csv_path: Path to the csv file

    returns_
        - List of subject names
        - List of clip names
        - List of camera names
        - List of frame numbers
        - Numpy.ndarray (num_images x 17 x 3) with the joint locations and the "visibility" flag.
    """
    with open(csv_path, "r") as f:
        first_line = f.readline()
        num_annotations = int(first_line.strip("#").strip("\n").strip("\r").strip(";"))

    offset = 4
    subjects = list()
    clips = list()
    cameras = list()
    frame_nums = list()
    joints = np.ndarray(shape=(num_annotations, num_joints, 3), dtype=float)

    with open(csv_path, "r") as f:
        reader = csv.reader(SkipCSVCommentsIterator(f), delimiter=';')
        row_count = 0
        skip_header = True
        for row in reader:
            if skip_header:
                skip_header = False
                continue
            assert (len(row) == offset + 3 * num_joints)
            subjects.append(row[0])
            clips.append(row[1])
            cameras.append(row[2])
            frame_nums.append(int(row[3]))
            joint_count = 0
            for i in range(offset + 2, len(row), 3):
                joints[row_count, joint_count] = [float(row[i - 2]), float(row[i - 1]), float(row[i])]
                joint_count += 1
            row_count += 1

    return subjects, clips, cameras, frame_nums, joints


def load_dataset_and_2d_poses(dataset_path, dataset_name="aspset", split="train", verbose=True, extract_info_from_vids=False, prediction_path=None):
    """
    Load VP3d-style 3D pose dataset (Human3.6m so far), along with fitting 2D poses
    :param dataset_path: Path to 3D dataset in .npz format
    :param poses_2d_path: Path to 2D poses in .npz format
    :param verbose: verbosity
    :return: dataset: MocapDataset, keypoints
    """
    if verbose:
        print(f'Loading 3D dataset from {dataset_path}')
    if dataset_name != "aspset":
        raise KeyError('Invalid dataset')

    if verbose:
        print("Converting 3D poses from world to camera frame and loading 2D poses")

    aspset = Aspset510(dataset_path)
    dataset = defaultdict(dict)
    keypoints = defaultdict(lambda: defaultdict(dict))

    if prediction_path is not None:
        pred_2d = read_aspset_csv_annotations(prediction_path.format(split))
        pred_2d = create_dictionary(pred_2d)

    cam_resolution = {}
    if os.path.exists(os.path.join(dataset_path, f"camera_resolution_{split}.json")) and not extract_info_from_vids:
        with open(os.path.join(dataset_path, f"camera_resolution_{split}.json"), 'r') as f:
            cam_resolution = json.load(f)

    for subject_id, clip_id in tqdm(aspset.splits[split]):
        clip = aspset.clip(subject_id, clip_id)
        anim = {}

        positions_3d = []
        for camera_num, camera_id in enumerate(clip.camera_ids):
            # 3d position in camera coordinates (revert translation and rotation)
            camera = clip.load_camera(camera_id)
            mocap = clip.load_mocap()
            skeleton = skeleton_registry[mocap.skeleton_name]
            anim['positions'] = mocap.joint_positions
            pos_3d = camera.world_to_camera_space(anim['positions'])[:, :, :3]
            pos_3d = pos_3d[:, APSSet17POrder.to_our_17p_order()]
            pos_3d /= 1000
            positions_3d.append(pos_3d)
            if "cameras" not in anim:
                anim["cameras"] = []

            joints_2d = to_cartesian(camera.world_to_image_space(anim['positions']))
            kps = joints_2d[:, APSSet17POrder.to_our_17p_order()].copy()
            if subject_id not in cam_resolution or clip_id not in cam_resolution[subject_id] or camera_id not in cam_resolution[subject_id][clip_id]:
                video_path = clip.get_video_path(camera_id)
                video = cv2.VideoCapture(str(video_path))
                fps = video.get(cv2.CAP_PROP_FPS)
                video.set(cv2.CAP_PROP_POS_FRAMES, 0)
                ret, image = video.read()
                if not ret:
                    raise RuntimeError(f'failed to read frame from video: {video_path}')
                height, width = image.shape[:2]
                if subject_id not in cam_resolution:
                    cam_resolution[subject_id] = {}
                if clip_id not in cam_resolution[subject_id]:
                    cam_resolution[subject_id][clip_id] = {}
                cam_resolution[subject_id][clip_id][camera_id] = (width, height, fps)
            else:
                width, height, fps = cam_resolution[subject_id][clip_id][camera_id]
            kps[..., :2] = normalize_screen_coordinates(kps[..., :2], w=width, h=height)
            if prediction_path is not None:
                predictions = [pred_2d[f"{subject_id};{clip_id};{camera_id};{frame}"] for frame in range(pos_3d.shape[0])]
                predictions = np.asarray(predictions)[:, :, :2]
                predictions = predictions[:, APSSet17POrder.to_our_17p_order()]
                predictions_norm = normalize_screen_coordinates(predictions, w=width, h=height)
                keypoints[subject_id][clip_id][camera_num] = predictions_norm
            else:
                keypoints[subject_id][clip_id][camera_num] = kps

            # Update camera to normalized 2d coordinates and meters instead of millimeters
            intrinsic = camera.intrinsic_matrix.copy()
            intrinsic[0:2, 2] = normalize_screen_coordinates(intrinsic[0:2, 2], w=width, h=height).astype(
                    'float32')  # center coordinate in (-1, 1) interval
            intrinsic[0, 0] = intrinsic[0, 0] / width * 2  # focal length matching image in (-1, 1) interval
            intrinsic[1, 1] = intrinsic[1, 1] / width * 2  # focal length matching image in (-1, 1) interval
            extrinsic = camera.extrinsic_matrix.copy()
            extrinsic[0:3, 3] = extrinsic[0:3, 3] / 1000  # mm to meters
            anim["cameras"].append((Camera(intrinsic, extrinsic), camera_id))

        anim['positions_3d'] = positions_3d
        dataset[subject_id][clip_id] = anim

    if not os.path.exists(os.path.join(dataset_path, f"camera_resolution_{split}.json")):
        with open(os.path.join(dataset_path, f"camera_resolution_{split}.json"), 'w') as f:
            json.dump(cam_resolution, f)

    return dataset, keypoints


def filter_and_subsample_dataset(dataset, poses_2d, downsample=1, verbose=True):

    out_poses_3d = []
    out_poses_2d = []
    out_camera_params = []
    out_subjects = []
    out_clips = []
    out_frame_rates = []
    out_cam_names = []

    # Mapping of names to indices
    subject_dict = {name: i for i, name in enumerate(dataset.keys())}
    clip_dict = {name: i for i, name in enumerate(np.concatenate([list(dataset[subject].keys()) for subject in dataset.keys()]))}

    for subject in dataset.keys():
        for clip in poses_2d[subject].keys():
            poses_2d_sequences = poses_2d[subject][clip]
            for i in range(len(poses_2d_sequences)):  # Iterate over cameras
                out_poses_2d.append(poses_2d_sequences[i].copy())
                out_subjects.append(subject_dict[subject])
                out_clips.append(clip_dict[clip])

            assert 'positions_3d' in dataset[subject][clip]
            cams = dataset[subject][clip]['cameras']
            for cam in cams:
                cam, cam_id = cam
                out_camera_params.append({"intrinsic" : cam.intrinsic_matrix.copy(),
                                          "extrinsic": cam.extrinsic_matrix.copy(),
                                          "name": cam_id})
            assert len(cams) == len(poses_2d_sequences), 'Camera count mismatch'
            poses_3d_sequences = dataset[subject][clip]['positions_3d']
            assert len(poses_3d_sequences) == len(poses_2d_sequences), 'Camera count mismatch'
            for i in range(len(poses_3d_sequences)):  # Iterate over cameras
                out_poses_3d.append(poses_3d_sequences[i].copy())
                if 'frame_rate' in dataset[subject][clip].keys():
                    frame_rate = dataset[subject][clip]['frame_rate']
                else:
                    frame_rate = 50
                out_frame_rates.append(frame_rate)

    if len(out_camera_params) == 0:
        out_camera_params = None
    if len(out_poses_3d) == 0:
        out_poses_3d = None
    if len(out_frame_rates) == 0:
        out_frame_rates = None

    if downsample > 1:
        # Downsample as requested
        for i in range(len(out_poses_2d)):
            out_poses_2d[i] = out_poses_2d[i][::downsample]
            if out_poses_3d is not None:
                out_poses_3d[i] = out_poses_3d[i][::downsample]
    # intrinsic camera parameters, 3d pose sequences in full length, 2d pose sequences in full length, frame names matching 2d poses, subject ids, action ids
    return out_camera_params, out_poses_3d, out_poses_2d, out_subjects, out_clips, out_frame_rates, (subject_dict, clip_dict)
