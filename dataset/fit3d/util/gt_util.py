# -*- coding: utf-8 -*-
"""
Created on 01.08.24

@author: Katja

"""
import os
import pickle

import torch
from tqdm import tqdm

import config
from dataset.fit3d.keypoint_order_fit3d import Fit3DOrder
from dataset.fit3d.util.dataset_util import read_data
from dataset.fit3d.util.smplx_util import SMPLXHelper


def load_fit3d_gt(data_path, split, camera_params=True, all_joints=False, use_nose=True):
    save_path = os.path.join(data_path, f"{split}_gt.pkl" if not camera_params else f"{split}_gt_camera.pkl")
    save_path = save_path.replace(".pkl", "_all.pkl") if all_joints else save_path
    save_path = save_path.replace(".pkl", "_37.pkl") if use_nose else save_path
    if not camera_params:
        print("Using WORLD coordinates")
    else:
        print("Using CAMERA coordinates")
    if os.path.exists(save_path):
        print("Using cached version from path ", save_path)
        with open(save_path, "rb") as f:
            return pickle.load(f)
    data_root = data_path[:data_path.rfind("fit3d")]
    gt = {}
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    smplx_helper = SMPLXHelper(config.SMPL_MODEL_DIR, load_renderer=False, device=device)
    subjects_val = ["s11"]
    betas_gt = {}
    gt_smplx = {}
    print(f"Loading {split} data")
    for subject_name in [f for f in os.listdir(os.path.join(data_root, "fit3d", "train")) if
                         os.path.isdir(os.path.join(data_root, "fit3d", "train", f))]:
        if split == "val" and subject_name not in subjects_val or split == "train" and subject_name in subjects_val:
            continue
        print(f"processing {subject_name}")
        gt[subject_name] = {}
        betas_gt[subject_name] = []
        gt_smplx[subject_name] = {}
        for cam_id, camera_name in enumerate([f for f in os.listdir(os.path.join(data_root, "fit3d", "train", subject_name, "videos")) if
                            os.path.isdir(os.path.join(data_root, "fit3d", "train", subject_name, "videos", f))]):
            gt[subject_name][camera_name] = {}
            gt_smplx[subject_name][camera_name] = {}
            for action_name in tqdm(
                    [f[:-4] for f in os.listdir(os.path.join(data_root, "fit3d", "train", subject_name, "videos", camera_name)) if
                     f.endswith(".mp4")]):
                _, _, cam_params, _, smplx_params, _ = read_data(data_root, "fit3d", "train", subject_name, action_name,
                                                                 camera_name, subject="w_markers",
                                                                 needed_vals=["cam_params", "smplx_params"])
                betas_gt[subject_name].extend(smplx_params["betas"])
                if not camera_params:
                    world_smplx_params = smplx_helper.get_world_smplx_params(smplx_params)
                    posed_data = smplx_helper.smplx_model(**world_smplx_params)
                else:
                    camera_smplx_params = smplx_helper.get_camera_smplx_params(smplx_params, cam_params)
                    posed_data = smplx_helper.smplx_model(**camera_smplx_params)
                joints_smplx = posed_data.joints.cpu().numpy()
                if not all_joints:
                    joints_smplx = joints_smplx[:, Fit3DOrder.from_SMPLX_order(use_nose=use_nose)]
                gt[subject_name][camera_name][action_name] = joints_smplx
                gt_smplx[subject_name][camera_name][action_name] = {i+1: {"betas": camera_smplx_params["betas"][i].cpu().numpy(),
                                                          "body_pose": camera_smplx_params["body_pose"][i].cpu().numpy(),
                                                          "global_orient": camera_smplx_params["global_orient"][i].cpu().numpy()} for i in range(len(camera_smplx_params["betas"]))}
    print(f"Finished computation, now saving to {save_path}")
    with open(save_path, "wb") as f:
        pickle.dump(gt, f)
    with open(save_path.replace(".pkl", "_betas.pkl"), "wb") as f:
        pickle.dump(betas_gt, f)
    with open(save_path.replace(".pkl", "_smplx_vals.pkl"), "wb") as f:
        pickle.dump(gt_smplx, f)
    return gt