# -*- coding: utf-8 -*-
"""
Created on 25.10.23



"""
import os

import math
import sys

import numpy as np
from torch.utils.data import Dataset, DataLoader

from dataset.camera import world_to_camera, normalize_screen_coordinates
from dataset.h36m import h36m_splits
from dataset.h36m.keypoint_order import H36MOrder17POriginalOrder
from dataset.mocap_dataset import MocapDataset


class H36mSequenceGenerator(Dataset):

    def __init__(self, dataset_path, dataset2d_path, subset, seq_len, subsample, stride=1, padding_type="zeros",
                 flip_augment=True, flip_lr_indices=None,
                 mask_stride=None, stride_mask_align_global=False, rand_shift_stride_mask=False, seed=0, verbose=True):

        # Build h36m dataset
        dataset_name = "h36m"
        selected_subjects = h36m_splits.subjects_by_split[subset]

        dataset_3d, poses_2d_dataset = load_dataset_and_2d_poses(dataset_path=dataset_path,
                                                                 poses_2d_path=dataset2d_path,
                                                                 dataset_name=dataset_name,
                                                                 verbose=True)

        action = "*"
        camera_params, poses_3d, poses_2d, _, subjects, actions, frame_rates, dicts = filter_and_subsample_dataset(
                dataset=dataset_3d,
                poses_2d=poses_2d_dataset,
                subjects=selected_subjects,
                action_filter=action,
                downsample=1,
                image_base_path=dataset_path,
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

        if do_flip:
            # Width (or x coord) is 0 centered, so flipping is simply sign inversion
            sequence_3d = sequence_3d[:, self.flip_lr_indices].copy()
            sequence_3d[..., 0] *= -1
            sequence_2d = sequence_2d[:, self.flip_lr_indices].copy()
            sequence_2d[..., 0] *= -1
            camera = camera.copy()
            # Flip cx (principal point)
            camera[4] *= -1
            # Flip t2 (tangential distortion)
            camera[9] *= -1

        return sequence_3d, sequence_2d, mask, camera, subject, action, i, stride_mask

    def flip_batch(self, sequence_3d, sequence_2d, camera):
        # Width (or x coord) is 0 centered, so flipping is simply sign inversion
        sequence_3d = sequence_3d[:, :, self.flip_lr_indices].clone()
        sequence_3d[..., 0] *= -1
        sequence_2d = sequence_2d[:, :, self.flip_lr_indices].clone()
        sequence_2d[..., 0] *= -1
        camera = camera.clone()
        # Flip cx (principal point)
        camera[4] *= -1
        # Flip t2 (tangential distortion)
        camera[9] *= -1

        return sequence_3d, sequence_2d, camera


def create_h36m_dataloaders(dataset_3d_path, dataset_2d_path, config, split, selection):

    if sys.gettrace() is None:
        WORKERS = 16
    else:
        WORKERS = 0

    subsample = config.DATASET_TRAIN_3D_SUBSAMPLE_STEP if split == "train" else config.DATASET_VAL_3D_SUBSAMPLE_STEP if (
            split == "val") else config.DATASET_TEST_3D_SUBSAMPLE_STEP
    shuffle = split == "train"
    stride_mask_rand_shift = (config.STRIDE_MASK_RAND_SHIFT and split == "train")
    subjects = h36m_splits.subjects_by_split[selection]
    actions = "*"
    do_flip = split == "train" and config.AUGM_FLIP_PROB > 0 and not config.IN_BATCH_AUGMENT

    dataset = H36mSequenceGenerator(dataset_path=dataset_3d_path,
                                    dataset2d_path=dataset_2d_path,
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
                                    seed=config.SEED)

    data_loader = DataLoader(
            dataset,
            batch_size=config.BATCH_SIZE // 2 if config.IN_BATCH_AUGMENT and split == "train" else config.BATCH_SIZE,
            shuffle=shuffle,
            num_workers=WORKERS,
            pin_memory=False,
            drop_last=shuffle
            )
    return dataset, data_loader


def create_h36m_datasets(dataset_3d_path, dataset_2d_path, config, train_subset, val_subset):
    train_dataset, val_dataset, val_batches, train_dataloader, val_dataloader = None, None, None, None, None

    for split, selection in zip(["train", "val"], [train_subset, val_subset]):
        if selection is not None:
            # The dataset is subsampled to every Nth frame (i.e. a sequence is extracted at every Nth frame)
            # The frame rate is not changed, however!
            dataset, data_loader = create_h36m_dataloaders(dataset_3d_path, dataset_2d_path, config, split, selection)

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


def load_dataset_and_2d_poses(dataset_path, poses_2d_path, dataset_name="h36m", verbose=True):
    """
    Load VP3d-style 3D pose dataset (Human3.6m so far), along with fitting 2D poses
    :param dataset_path: Path to 3D dataset in .npz format
    :param poses_2d_path: Path to 2D poses in .npz format
    :param verbose: verbosity
    :return: dataset: MocapDataset, keypoints
    """
    if verbose:
        print(f'Loading 3D dataset from {dataset_path}')
    if dataset_name == "h36m":
        from dataset.h36m.h36m_dataset import Human36mDataset
        dataset = Human36mDataset(dataset_path)
    else:
        raise KeyError('Invalid dataset')

    if verbose:
        print("Converting 3D poses from world to camera frame")
    for subject in list(dataset.subjects()):
        for action in dataset[subject].keys():
            anim = dataset[subject][action]

            if 'positions' in anim:
                positions_3d = []
                for cam in anim['cameras']:
                    # 3d position in camera coordinates (revert translation and rotation)
                    pos_3d = world_to_camera(anim['positions'], R=cam['orientation'], t=cam['translation'])
                    positions_3d.append(pos_3d)
                anim['positions_3d'] = positions_3d

    if verbose:
        print(f'Loading 2D poses from {poses_2d_path}')
    keypoints = np.load(poses_2d_path, allow_pickle=True)
    # keypoints_metadata = keypoints['metadata'].item()
    # keypoints_symmetry = keypoints_metadata['keypoints_symmetry']
    keypoints = keypoints['positions_2d'].item()

    for subject in dataset.subjects():
        assert subject in keypoints, 'Subject {} is missing from the 2D detections dataset'.format(subject)
        for action in dataset[subject].keys():
            assert action in keypoints[
                subject], 'Action {} of subject {} is missing from the 2D detections dataset'.format(action, subject)
            if 'positions_3d' not in dataset[subject][action]:
                continue

            for cam_idx in range(len(keypoints[subject][action])):  # for each action, there is one global 3D pose and 2D pose per camera

                # We check for >= instead of == because some videos in H3.6M contain extra frames
                mocap_length = dataset[subject][action]['positions_3d'][cam_idx].shape[0]
                assert keypoints[subject][action][cam_idx].shape[0] >= mocap_length

                if keypoints[subject][action][cam_idx].shape[0] > mocap_length:
                    # Shorten sequence
                    keypoints[subject][action][cam_idx] = keypoints[subject][action][cam_idx][:mocap_length]

            assert len(keypoints[subject][action]) == len(dataset[subject][action]['positions_3d'])

    if verbose:
        print("Normalizing 2D poses to [-1, 1] and converting to our 17-point order")
    for subject in keypoints.keys():
        for action in keypoints[subject]:
            for cam_idx, kps in enumerate(keypoints[subject][action]):
                # Normalize camera frame
                cam = dataset.cameras()[subject][cam_idx]
                # Convert default 17p order to our 17p order
                kps = kps[:, H36MOrder17POriginalOrder.to_our_17p_order()].copy()
                kps[..., :2] = normalize_screen_coordinates(kps[..., :2], w=cam['res_w'], h=cam['res_h'])
                keypoints[subject][action][cam_idx] = kps

    return dataset, keypoints


def filter_and_subsample_dataset(dataset: MocapDataset, poses_2d, subjects, action_filter, downsample=1, image_base_path=None,
                                 verbose=True):
    """
    Filter and subsample 3D pose dataset and corresponding 2d poses
    :param dataset: MocapDataset
    :param poses_2d: 2D poses
    :param subjects: list of subjects, e.g. ["S1", "S2"]
    :param action_filter: List of actions ["Walking", "Sitting"] or "*"
    :param downsample: downsample the framerate by this int-factor. defaults is 1 (no downsampling)
    :param image_base_path: Path to dataset, where the "frames" directory resides
    :param verbose: verbosity
    :return: poses_3d, poses_2d, camera_parameters (11 values), frame_names, each with one entry per sequence
    camera paramters are encoded as: res_w, res_h, fx, fy, cx, cy, r1, r2, r3, t1, t2, where f and c are normalized by res_w
    """
    if verbose is True:
        print(f"Filtering subjects: {subjects}")

    action_filter = None if action_filter == '*' else action_filter
    if action_filter is not None and verbose is True:
        print(f"Filtering actions: {action_filter}")

    translated_action_names = {"Photo":   "TakingPhoto",
                               "WalkDog": "WalkingDog"}

    out_poses_3d = []
    out_poses_2d = []
    out_camera_params = []
    out_frame_names = []
    out_subjects = []
    out_actions = []
    out_frame_rates = []

    splits = h36m_splits
    image_path_generator = h36m_splits.create_image_paths
    # Mapping of names to indices
    subject_dict = {name: i for i, name in enumerate(splits.all_subjects)}
    action_dict = {name: i for i, name in enumerate(splits.renamed_actions)}

    for subject in subjects:
        for action in poses_2d[subject].keys():
            action_name = action.split(' ')[0]
            if action_filter is not None:
                found = False
                for a in action_filter:
                    if action_name == a:
                        found = True
                        break
                if not found:
                    continue

            poses_2d_sequences = poses_2d[subject][action]
            for i in range(len(poses_2d_sequences)):  # Iterate over cameras
                out_poses_2d.append(poses_2d_sequences[i].copy())
                out_subjects.append(subject_dict[subject])
                out_actions.append(action_dict[action_name])

            if subject in dataset.cameras():
                cams = dataset.cameras()[subject]
                assert len(cams) == len(poses_2d_sequences), 'Camera count mismatch'
                for cam in cams:
                    if 'intrinsic' in cam:
                        out_camera_params.append(cam['intrinsic'].copy())

            if 'positions_3d' in dataset[subject][action]:
                poses_3d_sequences = dataset[subject][action]['positions_3d']
                assert len(poses_3d_sequences) == len(poses_2d_sequences), 'Camera count mismatch'
                for i in range(len(poses_3d_sequences)):  # Iterate over cameras
                    out_poses_3d.append(poses_3d_sequences[i].copy())
                    if 'frame_rate' in dataset[subject][action].keys():
                        frame_rate = dataset[subject][action]['frame_rate']
                    else:
                        frame_rate = 50
                    out_frame_rates.append(frame_rate)

            if image_base_path is not None:
                for i in range(len(poses_2d_sequences)):
                    num_frames = poses_2d_sequences[i].shape[0]
                    # 0-based frame names !!!
                    cam_id = dataset.cameras()[subject][i]['id']
                    frame_names = image_path_generator(image_base_path, subject, action, cam_id, range(num_frames))

                    # Revert canonical action renaming for correct frame names
                    original_name = None
                    new_name = None
                    for translated_name in translated_action_names.keys():
                        if translated_name in action:
                            new_name = translated_name
                            original_name = translated_action_names[translated_name]
                    if original_name is not None:
                        if not os.path.exists(frame_names[0]):
                            original_name = action.replace(new_name, original_name)
                            frame_names = image_path_generator(image_base_path, subject, original_name, cam_id,
                                                               range(num_frames))
                    # assert os.path.exists(frame_names[0])
                    out_frame_names.append(frame_names)

    if len(out_camera_params) == 0:
        out_camera_params = None
    if len(out_poses_3d) == 0:
        out_poses_3d = None
    if len(out_frame_names) == 0:
        out_frame_names = None
    if len(out_frame_rates) == 0:
        out_frame_rates = None

    if downsample > 1:
        # Downsample as requested
        for i in range(len(out_poses_2d)):
            out_poses_2d[i] = out_poses_2d[i][::downsample]
            if out_poses_3d is not None:
                out_poses_3d[i] = out_poses_3d[i][::downsample]
            if out_frame_names is not None:
                out_frame_names[i] = out_frame_names[i][::downsample]
    # intrinsic camera parameters, 3d pose sequences in full length, 2d pose sequences in full length, frame names matching 2d poses, subject ids, action ids
    return out_camera_params, out_poses_3d, out_poses_2d, out_frame_names, out_subjects, out_actions, out_frame_rates, (subject_dict, action_dict)
