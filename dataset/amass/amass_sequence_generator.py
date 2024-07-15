# -*- coding: utf-8 -*-
"""
Created on 25.10.23

@author: Katja

"""
import copy
import sys

import math

import numpy as np
import torch
from aspset510 import Aspset510, Camera
from glupy.math import to_cartesian
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from dataset.amass.amass_dataset import AMASSDataset
from dataset.camera import world_to_camera, project_to_2d, normalize_screen_coordinates
from dataset.h36m.h36m_dataset import Human36mDataset
from dataset.h36m.keypoint_order import H36MOrder17P
from dataset.uplifiting_dataset import world_to_cam_and_2d


class AMASSSequenceGenerator(Dataset):

    def __init__(self, amass_dataset: AMASSDataset, seq_len, target_frame_rate=50, subsample=1, stride=1, padding_type="zeros",
                 flip_augment=True, in_batch_augment=False, flip_lr_indices=None,
                 mask_stride=None, stride_mask_align_global=False, rand_shift_stride_mask=False,
                 shuffle=True, seed=0, verbose=True):
        """

        :param amass_dataset:
        :param seq_len:
        :param subsample:
        :param stride:
        :param padding_type:
        :param flip_augment:
        :param flip_lr_indices:
        :param shuffle:
        :param seed:
        :param verbose:
        """
        self.seq_len = seq_len
        self.subsample = subsample
        self.stride = stride
        self.target_frame_rate = target_frame_rate
        if padding_type == "zeros":
            self.pad_type = "constant"
        elif padding_type == "copy":
            self.pad_type = "edge"
        else:
            raise ValueError(f"Padding type not supported: {padding_type}")
        self.flip_augment = flip_augment
        self.in_batch_augment = in_batch_augment
        self.flip_lr_indices = flip_lr_indices
        self.abs_mask_stride = mask_stride
        if self.abs_mask_stride is not None:
            if type(self.abs_mask_stride) is not list:
                self.abs_mask_stride = [self.abs_mask_stride]
            for ams in self.abs_mask_stride:
                assert ams >= self.stride
                assert ams % self.stride == 0
        self.stride_mask_align_global = stride_mask_align_global
        self.rand_shift_stride_mask = rand_shift_stride_mask
        if self.rand_shift_stride_mask is True:
            assert self.stride_mask_align_global is False
        self.split = amass_dataset.split
        self.shuffle = shuffle
        self.seed = seed
        self.rng = np.random.default_rng(seed=self.seed)
        self.stride_shift_rng = np.random.default_rng(seed=self.seed)
        self.mask_stride_rng = np.random.default_rng(seed=self.seed)
        self.verbose = verbose

        if self.verbose:
            print("Generating sequences ...")

        if self.flip_augment is True:
            assert flip_lr_indices is not None

        # Extract all sequences from amass dataset
        self.sequences = []
        self.frame_rates = []

        # self.sequence_names = []
        for dataset, subjects in amass_dataset._data.items():
            for subject, actions in subjects.items():
                for action, sequence_3d in actions.items():
                    self.sequences.append(sequence_3d["positions"])
                    if 'frame_rate' in sequence_3d.keys():
                        frame_rate = sequence_3d['frame_rate']
                    else:
                        frame_rate = 50
                    self.frame_rates.append(frame_rate)
                    # self.sequence_names.append(
                    #     f"{sequence_3d['dataset']}/{sequence_3d['subject']}/{sequence_3d['action']}")

        # Extract all cameras
        self.cameras = []
        if amass_dataset.camera_type == "h36m":
            for subject, cams in amass_dataset.cameras().items():
                for cam in cams:
                    if "orientation" in cam.keys():
                        rot = cam["orientation"]  # 4 values
                        translation = cam["translation"]  # 3 values
                        intrinsics = cam["intrinsic"]  # 2+2+2+3+2 = 11 values
                        cam_params = np.concatenate([rot, translation, intrinsics], axis=0).astype(np.float32)
                        self.cameras.append(cam_params)
        else:
            for i in range(amass_dataset.cameras().shape[0]):
                cam = amass_dataset.cameras()[i]
                extrinsic_matrix = cam[:4]
                intrinsic_matrix = cam[4:]
                intrinsic = intrinsic_matrix.copy()
                intrinsic[0:2, 2] = normalize_screen_coordinates(intrinsic[0:2, 2], w=3840, h=2160).astype(
                        'float32')  # center coordinate in (-1, 1) interval
                intrinsic[0, 0] = intrinsic[0, 0] / 3840 * 2  # focal length matching image in (-1, 1) interval
                intrinsic[1, 1] = intrinsic[1, 1] / 3840 * 2  # focal length matching image in (-1, 1) interval
                extrinsic = extrinsic_matrix.copy()
                extrinsic[0:3, 3] = extrinsic[0:3, 3] / 1000  # mm to meters
                camera = Camera(intrinsic, extrinsic)
                self.cameras.append(camera)


        # Generate all central locations of sequences.
        self.sequence_locations = []
        for s_i, sequence in enumerate(self.sequences):
            positions = np.arange(start=0, stop=sequence.shape[0], step=self.subsample)
            sequence_number = np.tile([s_i], reps=(positions.shape[0]))
            frame_rates_tiled = np.tile([self.frame_rates[s_i]], reps=(positions.shape[0]))
            do_flip = np.zeros(shape=(positions.shape[0]), dtype=positions.dtype)
            if self.flip_augment and not self.in_batch_augment:
                sequence_number = np.concatenate([sequence_number, sequence_number], axis=0)
                frame_rates_tiled = np.concatenate([frame_rates_tiled, frame_rates_tiled], axis=0)
                positions = np.concatenate([positions, positions], axis=0)
                do_flip = np.concatenate([do_flip, 1 - do_flip], axis=0)

            self.sequence_locations.append(np.stack([sequence_number, positions, do_flip, frame_rates_tiled], axis=-1))

        self.sequence_locations = np.concatenate(self.sequence_locations, axis=0)

    def __len__(self):
        return len(self.sequence_locations)

    def __getitem__(self, item):

        # ToDo: For compatibility, return a dummy subject and action. Might change in the future
        subject, action = 0, 0

        s_i, i, do_flip, frame_rate = self.sequence_locations[item]
        frame_rate = int(frame_rate)
        stride = self.stride
        mult = 1
        assert frame_rate % self.target_frame_rate == 0
        if frame_rate != self.target_frame_rate:
            mult = frame_rate // self.target_frame_rate
            stride *= mult

        if self.abs_mask_stride is None:
            abs_mask_stride = stride
        else:
            if len(self.abs_mask_stride) == 1:
                abs_mask_stride = self.abs_mask_stride[0]
            else:
                abs_mask_stride = self.abs_mask_stride[
                    self.mask_stride_rng.integers(low=0, high=len(self.abs_mask_stride), endpoint=False)]
            abs_mask_stride *= mult

        mask_stride = abs_mask_stride // stride

        left = (self.seq_len - 1) * stride // 2
        right = (self.seq_len - 1) * stride - left

        do_flip = do_flip == 1.
        video_sequence = self.sequences[s_i]
        video_len = video_sequence.shape[0]
        begin, end = i - left, i + right + 1
        pad_left, pad_right = 0, 0
        if begin < 0:
            pad_left = math.ceil(-begin / stride)
            last_pad = begin + ((pad_left - 1) * stride)
            begin = last_pad + stride
        if end > video_len:
            pad_right = math.ceil((end - video_len) / stride)
            first_pad = end - ((pad_right - 1) * stride)
            end = first_pad - stride

        # Base case:
        sequence_3d = video_sequence[begin: end: stride]
        mask = np.ones(sequence_3d.shape[0], dtype=np.float32)
        # Pad if necessary
        if pad_left > 0 or pad_right > 0:
            # numpy constant padding defaults to 0 values
            sequence_3d = np.pad(sequence_3d, ((pad_left, pad_right), (0, 0), (0, 0)), mode=self.pad_type)
            mask = np.pad(mask, (pad_left, pad_right), mode="constant")

        # Generate stride mask that is centered on the central frame
        mid_index = self.seq_len // 2
        sequence_indices = np.arange(0, self.seq_len) - mid_index
        sequence_indices *= stride
        if self.stride_mask_align_global is True:
            # Shift mask such that it is aligned on the global frame indices
            # This is required for inference mode
            sequence_indices += i

        elif self.rand_shift_stride_mask is True:
            # Shift stride mask randomly by [ceil(-mask_stride/2), floor(mask_stride/2)]
            max_shift = int(np.ceil((mask_stride - 1) / 2))
            endpoint = mask_stride % 2 != 0
            rand_shift = self.stride_shift_rng.integers(low=-max_shift, high=max_shift, endpoint=endpoint)
            rand_shift *= stride
            sequence_indices += rand_shift

        stride_mask = np.equal(sequence_indices % abs_mask_stride, 0)

        assert sequence_3d.shape[0] == self.seq_len
        assert mask.shape[0] == self.seq_len
        assert stride_mask.shape[0] == self.seq_len

        # Ok, so:
        # Using a randomly selected H36m cameras to project AMASS sequences will at times lead to 2D poses
        # outside the target range of [-1, 1] in height or width, i.e. outside the actual sensor
        # In the train split, this happens in about 2% of cases (given a fixed seed for random camera selection)
        # and about 5% of cases in the val split
        # Now, we could try to project to 2D right here and re-draw a camera until we have a projection in the target range,
        # but we have implemented 2D projection as part of the tf dataset pipeline.
        # Thus, for now, we will simply accept the cases of 2D poses outside the target range,
        # but this will hopefully not change much. We simply emulate a larger sensor size.
        cam = self.cameras[self.rng.integers(low=0, high=len(self.cameras), size=1)[0]]

        if do_flip is True:
            # Width (or x coord) is 0 centered, so flipping is simply sign inversion
            sequence_3d = sequence_3d[:, self.flip_lr_indices].copy()
            sequence_3d[..., 0] *= -1
            # We do not alter the camera here
            # Flipping only changes the pose sequence along the x axis, but not the cameras

        return sequence_3d, np.empty(0), cam, mask, subject, action, i, stride_mask

    def flip_batch(self, sequence_3d):
        # Width (or x coord) is 0 centered, so flipping is simply sign inversion
        sequence_3d = sequence_3d[:, :, self.flip_lr_indices].clone()
        sequence_3d[..., 0] *= -1

        return sequence_3d


class AMASSSequenceGenerator2D(Dataset):

    def __init__(self, amass_seq_gen: AMASSSequenceGenerator, in_batch_augment, camera_type="h36m"):
        self.amass_seq_gen = amass_seq_gen
        self.in_batch_augment = in_batch_augment
        self.camera_type = camera_type

    def __len__(self):
        return len(self.amass_seq_gen)

    def __getitem__(self, item):
        sequence_3d, _, cam, mask, subject, action, i, stride_mask = self.amass_seq_gen[item]
        sequence_3d = torch.from_numpy(sequence_3d).float()
        if self.camera_type == "h36m":
            cam = torch.from_numpy(cam).float()
            sequence_3d_cam, sequence_2d = world_to_cam_and_2d(sequence_3d, cam)
            if self.in_batch_augment:
                sequence_3d_flipped = sequence_3d[:, self.amass_seq_gen.flip_lr_indices].clone()
                sequence_3d_flipped[..., 0] *= -1
                sequence_3d_cam_flipped, sequence_2d_flipped = world_to_cam_and_2d(sequence_3d_flipped, cam)
                sequence_3d_cam = torch.stack([sequence_3d_cam, sequence_3d_cam_flipped], dim=0)
                sequence_2d = torch.stack([sequence_2d, sequence_2d_flipped], dim=0)
        else:
            sequence_3d_cam = cam.world_to_image_space(sequence_3d)
            sequence_2d = to_cartesian(sequence_3d_cam)
            if self.in_batch_augment:
                sequence_3d_flipped = sequence_3d[:, self.amass_seq_gen.flip_lr_indices].clone()
                sequence_3d_flipped[..., 0] *= -1
                sequence_3d_cam_flipped = cam.world_to_image_space(sequence_3d_flipped)
                sequence_2d_flipped = to_cartesian(sequence_3d_cam_flipped)
                sequence_3d_cam = torch.stack([sequence_3d_cam, sequence_3d_cam_flipped], dim=0)
                sequence_2d = torch.stack([sequence_2d, sequence_2d_flipped], dim=0)
            cam = np.concatenate([cam.extrinsic_matrix, cam.intrinsic_matrix], axis=0)
        return sequence_3d_cam, sequence_2d, mask, cam, subject, action, i, stride_mask


def create_amass_h36m_datasets(amass_path, h36m_path, config, train_subset, val_subset,
                               target_frame_rate):
    # Build amass h36m dataset
    cam_dataset = Human36mDataset(h36m_path)
    cameras = copy.deepcopy(cam_dataset.cameras())
    del (cam_dataset)

    train_dataset, val_dataset, val_batches, train_dataloader, val_dataloader = None, None, None, None, None
    for split, selection in zip(["train", "val"], [train_subset, val_subset]):
        if selection is not None:
            print(f"Loading AMASS dataset for split {selection}")
            amass_dataset = AMASSDataset(path=amass_path, split=selection,
                                         cameras=cameras)

            # The dataset is subsampled to every Nth frame (i.e. a sequence is extracted at every Nth frame)
            # The frame rate is not changed, however!
            stride = config.DATASET_TRAIN_3D_SUBSAMPLE_STEP if split == "train" else config.DATASET_VAL_3D_SUBSAMPLE_STEP
            shuffle = split == "train"
            stride_mask_rand_shift = config.STRIDE_MASK_RAND_SHIFT and split == "train"
            do_flip = split == "train" and config.AUGM_FLIP_PROB > 0
            dataset = AMASSSequenceGenerator(amass_dataset=amass_dataset,
                                               seq_len=config.SEQUENCE_LENGTH,
                                               target_frame_rate=target_frame_rate,
                                               subsample=stride,
                                               stride=config.SEQUENCE_STRIDE,
                                               padding_type=config.PADDING_TYPE,
                                               flip_augment=do_flip,
                                               in_batch_augment=config.IN_BATCH_AUGMENT,
                                               flip_lr_indices=H36MOrder17P.flip_lr_indices(),
                                               mask_stride=config.MASK_STRIDE,
                                               stride_mask_align_global=False,
                                               rand_shift_stride_mask=stride_mask_rand_shift,
                                               seed=config.SEED)

            dataset = AMASSSequenceGenerator2D(amass_seq_gen=dataset, in_batch_augment=config.IN_BATCH_AUGMENT and split == "train",
                                               camera_type="h36m")
            print(f"Sequences: {len(dataset)}")

            if sys.gettrace() is None:
                WORKERS = 16
            else:
                WORKERS = 0

            data_loader = DataLoader(
                    dataset,
                    batch_size=config.BATCH_SIZE // 2 if config.IN_BATCH_AUGMENT and split == "train" else config.BATCH_SIZE,
                    shuffle=shuffle,
                    num_workers=WORKERS,
                    pin_memory=False,
                    drop_last=shuffle
                    )

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


def create_amass_aspset_datasets(amass_path, aspset_path, config, train_subset, val_subset,
                               target_frame_rate):
    # Build amass h36m dataset
    cam_dataset = Aspset510(aspset_path)
    cameras = []

    for subject_id, clip_id in tqdm(cam_dataset.splits["train"]):
        clip = cam_dataset.clip(subject_id, clip_id)
        for camera_num, camera_id in enumerate(clip.camera_ids):
            camera = clip.load_camera(camera_id)
            cameras.append(np.concatenate([camera.extrinsic_matrix, camera.intrinsic_matrix], axis=0))

    cameras = np.asarray(cameras)
    cameras = np.unique(cameras, axis=0)
    del (cam_dataset)

    train_dataset, val_dataset, val_batches, train_dataloader, val_dataloader = None, None, None, None, None
    for split, selection in zip(["train", "val"], [train_subset, val_subset]):
        if selection is not None:
            print(f"Loading AMASS dataset for split {selection}")
            amass_dataset = AMASSDataset(path=amass_path, split=selection,
                                         cameras=cameras, camera_type="aspset")

            # The dataset is subsampled to every Nth frame (i.e. a sequence is extracted at every Nth frame)
            # The frame rate is not changed, however!
            stride = config.DATASET_TRAIN_3D_SUBSAMPLE_STEP if split == "train" else config.DATASET_VAL_3D_SUBSAMPLE_STEP
            shuffle = split == "train"
            stride_mask_rand_shift = config.STRIDE_MASK_RAND_SHIFT and split == "train"
            do_flip = split == "train" and config.AUGM_FLIP_PROB > 0
            dataset = AMASSSequenceGenerator(amass_dataset=amass_dataset,
                                               seq_len=config.SEQUENCE_LENGTH,
                                               target_frame_rate=target_frame_rate,
                                               subsample=stride,
                                               stride=config.SEQUENCE_STRIDE,
                                               padding_type=config.PADDING_TYPE,
                                               flip_augment=do_flip,
                                               in_batch_augment=config.IN_BATCH_AUGMENT,
                                               flip_lr_indices=H36MOrder17P.flip_lr_indices(),
                                               mask_stride=config.MASK_STRIDE,
                                               stride_mask_align_global=False,
                                               rand_shift_stride_mask=stride_mask_rand_shift,
                                               seed=config.SEED)

            dataset = AMASSSequenceGenerator2D(amass_seq_gen=dataset, in_batch_augment=config.IN_BATCH_AUGMENT and split == "train",
                                               camera_type="aspset")
            print(f"Sequences: {len(dataset)}")

            if sys.gettrace() is None:
                WORKERS = 16
            else:
                WORKERS = 0

            data_loader = DataLoader(
                    dataset,
                    batch_size=config.BATCH_SIZE // 2 if config.IN_BATCH_AUGMENT and split == "train" else config.BATCH_SIZE,
                    shuffle=shuffle,
                    num_workers=WORKERS,
                    pin_memory=False,
                    drop_last=shuffle
                    )

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