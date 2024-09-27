# -*- coding: utf-8 -*-
"""
Created on 09.08.24

@author: Katja

"""
import os

import einops
import numpy as np

from config import ASPSET_REGRESSOR_PATH
from dataset.aspset.util import load_aspset_gt


def run_on_aspset(split, data_dir, save_dir, gender, num_procs):

    if not data_dir.endswith(".pkl"):
        source_pts_dict = load_aspset_gt(data_dir, split)
    else:
        source_pts_dict = load_uu_results(data_dir)

    already_processed = [f for f in os.listdir(save_dir) if f.endswith(".npz")]

    regressor = np.load(ASPSET_REGRESSOR_PATH)

    for i, (s, vals1) in enumerate(source_pts_dict.items()):
        for c, vals2 in vals1.items():
            for cam, joints3d in vals2.items():
                if f"{s}_{c}_{cam}_ik.npz" in already_processed:
                    continue
                print(f"Processing {s} {c} {cam}")
                root = joints3d[:, H36MOrder17P.pelvis, :3]
                verts = joints3d[:, :, :3] - einops.repeat(joints3d[:, H36MOrder17P.pelvis, :3], 'm n -> m k n', k=17)
                verts = verts.reshape(-1, 3)
                rot_mat = np.asarray([[1, 0, 0], [0, 0, 1], [0, -1, 0]])
                verts = (rot_mat @ verts.T).T
                verts = verts.reshape(-1, 17, 3)
                run_ik_fitting(joints=verts,
                               keypoint_order=None,
                               save_path=os.path.join(save_dir, f"{s}_{c}_{cam}_ik.npz"),
                               translation=root,
                               gender=gender,
                               regressor=regressor,
                               num_processes_per_gpu=num_procs)

if __name__ == '__main__':
    import argparse

    # Parse command line arguments
    parser = argparse.ArgumentParser(
            description='IK on ASPset.')
    parser.add_argument('--data_path', required=True,
                        help="dataset path, either path to GT data or the pkl file with results from Uplift Upsample, generated with the eval script.")
    parser.add_argument('--gpus', required=True,
                        default=None,
                        help="GPUs to use, e.g., '0, 1, 2'")
    parser.add_argument('--num_procs', required=False,
                        default=8,
                        metavar="num_processes_per_gpu",
                        help='Number of processes per GPU',
                        type=int)
    parser.add_argument('--save_path', required=False,
                        default=os.path.join(".", "inverse_kinematics", "results_aspset"),
                        help="path to save results")
    parser.add_argument('--gender', required=False,
                        default="neutral",
                        help="gender of SMPL-X model")
    parser.add_argument('--split', required=False,
                        default="test",
                        help="split of ASPset data")

    args = parser.parse_args()
    # set the GPUs you want to use
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus

    print(f"Running on GPU {os.environ['CUDA_VISIBLE_DEVICES']}")

    from inverse_kinematics.ik_joints import run_ik_fitting
    from dataset.h36m.keypoint_order import H36MOrder17P
    from uplift_upsample.utils.load_results import load_uu_results

    save_dir = os.path.join(args.save_path,  f"{args.split}_{args.gender}")
    os.makedirs(save_dir, exist_ok=True)
    run_on_aspset(args.split, args.data_path, save_dir, args.gender)