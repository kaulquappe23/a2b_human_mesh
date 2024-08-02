# -*- coding: utf-8 -*-
"""
Created on 02.08.24

@author: Katja

"""
import os

from smplx.lbs import batch_rodrigues

import config

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from smplx import build_layer

from dataset.fit3d.util.smplx_util import smplx_cfg

import torch
import torch.multiprocessing as mp

mp.set_start_method('spawn', force=True)

from multiprocessing import  Manager

import numpy as np

from torch import nn

from human_body_prior.models.ik_engine import IK_Engine


class IK_Keypoints(nn.Module):
    """

    """
    def __init__(self, used_joints=None):
        super(IK_Keypoints, self).__init__()

        self.bm = build_layer(config.SMPL_MODEL_DIR, **smplx_cfg)
        self.used_joints = used_joints

    def forward(self, body_parms):

        global_orient = batch_rodrigues(body_parms["root_orient"])
        body_pose = body_parms["pose_body"].reshape(21, 3)
        human = self.bm(betas=body_parms["betas"], return_verts=True,
                        body_pose=batch_rodrigues(body_pose)[None],
                        global_orient=global_orient,
                        transl=body_parms["trans"]
                        )
        if self.used_joints is not None:
            joints = human.joints[:, self.used_joints]
        else:
            joints = human.joints
        return {'source_kpts': joints, 'body': human}

def run_ik_fitting(joints, save_path, keypoint_order, translation=None, num_betas=10, num_processes_per_gpu=8):
    """
    Fit a SMPLX model to the given joints and save the result.
    Uses all available GPUs and splits the frames to process among them.
    :param num_processes_per_gpu: Number of processes to run per GPU. The frames will be split among all processes (num_gpus * num_processes_per_gpu)
    :param joints: N x K x 3 numpy array of 3D joints, K must match len(keypoint_order)
    :param save_path: File to save results
    :param translation: Translation that has to be applied to the joints after the fitting, used if the joints where normalized before fitting
    :param num_betas: Number of beta parameters of the SMPL-X model to fit
    :param keypoint_order: Mapping of SMPL-X joints to the joints in the input array -> list of indices of SMPL-X joints in the order of the joints from the input array
    :return:
    """
    num_gpus = sum(c.isdigit() for c in os.environ["CUDA_VISIBLE_DEVICES"])
    num_processes = num_processes_per_gpu * num_gpus
    if isinstance(joints, np.ndarray):
        joints = torch.tensor(joints).float()
    frame_ids = np.arange(joints.shape[0])
    with Manager() as  manager:
        final_results = manager.dict()
        final_results["trans"] = manager.dict()
        final_results["betas"] = manager.dict()
        final_results["root_orient"] = manager.dict()
        final_results["pose_body"] = manager.dict()

        frame_chunks = np.array_split(frame_ids, num_processes)
        joints_chunks = [joints[frame_chunks[i]] for i in range(num_processes)]

        jobs = []
        for i in range(num_processes):
            fs, target_pts = frame_chunks[i], joints_chunks[i]
            p = mp.Process(target=ik_fitting_worker, args=(fs, target_pts, final_results, keypoint_order, num_betas, i))
            jobs.append(p)
            p.start()

        for proc in jobs:
            proc.join()

        results = {}
        for k in final_results.keys():
            keys = list(final_results[k].keys())
            assert max(keys) == len(keys) - 1
            arr = [final_results[k][i] for i in range(len(keys))]
            res = np.concatenate(arr, axis=0)
            results[k] = res
        results["target_joints"] = joints.cpu().numpy()
        if translation is not None:
            results["translation"] = translation
    print(f"Saving to {save_path}")
    np.savez(save_path, **results)


def ik_fitting_worker(frame_ids, targets, final_results, keypoint_order, num_betas=10, worker_id=0):
    """
    One worker for the ik fitting
    :param frame_ids: frames to process
    :param targets: target 3D joints
    :param final_results: dictionary to store the results, shared among all processes
    :param num_betas: Number of beta parameters of the SMPL-X model to fit
    :param keypoint_order: Mapping of SMPL-X joints to the joints in the input array -> list of indices of SMPL-X joints in the order of the joints from the input array
    :param worker_id: id of the worker process
    :return:
    """
    num_gpus = sum(c.isdigit() for c in os.environ["CUDA_VISIBLE_DEVICES"])
    cuda_id = worker_id % num_gpus
    if not cuda_id >= 0 or not cuda_id < num_gpus:
        cuda_id = 0
    device = torch.device(f'cuda:{cuda_id}') if torch.cuda.is_available() else torch.device('cpu')

    ik_engine = IK_Engine(vposer_expr_dir=config.VPOSER_DIR,
                          verbosity=0,
                          display_rc=(2, 2),
                          data_loss=torch.nn.MSELoss(reduction='sum'),
                          stepwise_weights=[{'data': 10., 'poZ_body': .0007, 'betas': .01}, ],
                          optimizer_args={
                                  'type':         'LBFGS', 'max_iter': 300, 'lr': 1, 'tolerance_change': 1e-4,
                                  'history_size': 200
                                  },
                          num_betas=num_betas).to(device)

    for i, frame in enumerate(frame_ids):

        target_pts = targets[i].detach().to(device)[None]
        curr_ik_pts = IK_Keypoints(keypoint_order).to(device)

        # speed up process and enhance results: start from estimation from previous frame instead of T-Pose
        if frame-1 in final_results["betas"]:
            initial_body_params = {k: torch.from_numpy(v[frame-1]).to(device) for k, v in final_results.items()}
        else:
            initial_body_params = {}
        ik_res = ik_engine(curr_ik_pts, target_pts, initial_body_params=initial_body_params)

        ik_res_detached = {k: v.detach() for k, v in
                           ik_res.items()}  # returns trans, betas, root_orient, poZ_body, pose_body"

        for k in final_results.keys():
            final_results[k][frame] = ik_res_detached[k].cpu().numpy()