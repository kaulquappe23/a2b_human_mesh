# -*- coding: utf-8 -*-
"""
Created on 12.08.24

@author: Katja

"""
import os
import pickle

import numpy as np
from tqdm import tqdm


def create_ik_results(origin_dir, save_path):
    """
    Creates a pickle file with the results of the inverse kinematics run. This format is necessary for the evaluation
    :param origin_dir: Directory where the raw IK results are stored
    :param save_path: Base path for the new format. Two files will be generated: <save_path>_smplx_vals.pkl and <save_path>_res_betas.pkl
    :return:
    """

    dest_file = save_path + "_smplx_vals.pkl"
    dest_file_betas = save_path + "_res_betas.pkl"

    results = {}
    ik_beta_results = {"1e28": [], "8a59": []}
    for file in tqdm(os.listdir(origin_dir)):
        if not file.endswith(".npz"):
            continue
        s, c, cam = file[:-7].split("_")
        if s not in results:
            results[s] = {}
        if c not in results[s]:
            results[s][c] = {}
        results[s][c][cam] = {}
        sample = np.load(os.path.join(origin_dir, file))
        if s not in ik_beta_results:
            ik_beta_results[s] = []
        ik_beta_results[s].append(sample["betas"])
        for i in range(sample["betas"].shape[0]):
            results[s][c][cam][i+1] = {"betas": sample["betas"][i][None],
                                      "body_pose": sample["pose_body"][i][None],
                                      "global_orient": sample["root_orient"][i][None],
                                      "translation": sample["trans"][i][None]}

    for s in ik_beta_results:
        ik_beta_results[s] = np.concatenate(ik_beta_results[s])

    with open(dest_file, "wb") as f:
        pickle.dump(results, f)
        print(f"Wrote SMPL-X parameters to {dest_file}")
    with open(dest_file_betas, "wb") as f:
        pickle.dump(ik_beta_results, f)
        print(f"Wrote betas to {dest_file_betas}")



if __name__ == '__main__':
    import argparse

    # Parse command line arguments
    parser = argparse.ArgumentParser(
            description='IK evaluation.')
    parser.add_argument('--input', required=True,
                        help="Directory where the raw IK results are stored")
    parser.add_argument('--output', required=True,
                        help="Base path for the new format. Two files will be generated: <save_path>_smplx_vals.pkl and <save_path>_res_betas.pkl.")

    args = parser.parse_args()
    create_ik_results(args.input, args.output)