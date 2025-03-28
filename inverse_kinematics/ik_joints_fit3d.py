# -*- coding: utf-8 -*-
#
#
import os

import einops


def run_on_fit3d(split, subject_val, data_dir, save_dir, gender, num_procs):

    if not data_dir.endswith(".pkl"):
        source_pts_dict = load_fit3d_gt(data_dir, split, subject_val)
    else:
        source_pts_dict = load_uu_results(data_dir)

    already_processed = [f for f in os.listdir(save_dir) if f.endswith(".npz")]

    for i, (s, vals1) in enumerate(source_pts_dict.items()):
        for c, vals2 in vals1.items():
            for a, joints3d in vals2.items():
                if f"{s}_{c}_{a}_ik.npz" in already_processed:
                    continue
                print(f"Processing {s} {c} {a}")
                root = joints3d[:, 0, :3]
                verts = joints3d[:, :, :3] - einops.repeat(joints3d[:, 0, :3], 'm n -> m k n', k=joints3d.shape[1])
                run_ik_fitting(joints=verts,
                               keypoint_order=Fit3DOrder.from_SMPLX_order(),
                               save_path=os.path.join(save_dir, f"{s}_{c}_{a}_ik.npz"),
                               translation=root,
                               gender=gender,
                               num_processes_per_gpu=num_procs)


if __name__ == '__main__':
    import argparse

    # Parse command line arguments
    parser = argparse.ArgumentParser(
            description='IK on fit3D.')
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
                        default=os.path.join(".", "inverse_kinematics", "results_fit3d"),
                        help="path to save results")
    parser.add_argument('--gender', required=False,
                        default="neutral",
                        help="gender of SMPL-X model")
    parser.add_argument('--split', required=False,
                        default="val",
                        help="split of fit3D data")
    parser.add_argument('--subject_val', required=False,
                        default="s11",
                        help="subject in the validation set")

    args = parser.parse_args()
    # set the GPUs you want to use
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus

    print(f"Running on GPU {os.environ['CUDA_VISIBLE_DEVICES']}")

    from dataset.fit3d.util.gt_util import load_fit3d_gt
    from uplift_upsample.utils.load_results import load_uu_results
    from inverse_kinematics.ik_joints import run_ik_fitting
    from dataset.fit3d.keypoint_order_fit3d import Fit3DOrder

    save_dir = os.path.join(args.save_path,  f"{args.split}_{args.gender}")
    os.makedirs(save_dir, exist_ok=True)
    run_on_fit3d(args.split, args.subject_val, args.data_path, save_dir, args.gender, args.num_procs)
