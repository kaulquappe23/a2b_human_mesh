# -*- coding: utf-8 -*-
"""
Created on 12.03.24

@author: Katja

"""
import os.path
import pickle

import numpy as np
import smplx
import torch
from tqdm import tqdm

from config import SMPL_MODEL_DIR, ASPSET_REGRESSOR_PATH
from dataset.action_wise_eval import frame_wise_eval, print_results
from dataset.aspset.util import load_aspset_gt
from dataset.h36m.keypoint_order import H36MOrder17P
from evaluation.general_eval_utils import get_all_beta_params


def evaluate_results(aspset_path, prediction_dict, ignore_not_predicted=False, return_metrics_per_frame=False, model_name=None, verbose=False, split="test"):

    aspset_gt = load_aspset_gt(aspset_path, split)
    not_found = {}
    np_gt = []
    np_pred = []
    identifiers = []
    for s in aspset_gt:
        for c in aspset_gt[s]:
            for cam in aspset_gt[s][c]:
                a = aspset_gt[s][c][cam]
                p = np.asarray(prediction_dict[s][c][cam])
                if ignore_not_predicted:
                    check_not_predicted_frames = np.sum(p, axis=(1, 2)) == 0
                    if np.sum(check_not_predicted_frames) > 0:
                        if s not in not_found:
                            not_found[s] = {}
                        if c not in not_found[s]:
                            not_found[s][c] = {}
                        not_found[s][c][cam] = np.where(check_not_predicted_frames)[0]
                if a.shape[0] != p.shape[0]:
                    min_frames = min(a.shape[0], p.shape[0])
                    max_frames = max(a.shape[0], p.shape[0])
                    if verbose:
                        print(
                            f"Different number of frames in GT and prediction for {s} {c} {cam}! Missing {max_frames - min_frames} frames.")
                    a = a[:min_frames]
                    p = p[:min_frames]

                if s in not_found and c in not_found[s] and cam in not_found[s][c]:
                    frames_not_found = not_found[s][c][cam]
                    use_fames = [i for i in range(a.shape[0]) if i not in frames_not_found]
                else:
                    use_fames = np.arange(a.shape[0])
                ids = [f"{s}_{c}_{cam}_{i}" for i in use_fames]
                identifiers.extend(ids)
                np_gt.append(a[use_fames])
                np_pred.append(p[use_fames])

    np_gt = np.concatenate(np_gt)
    np_pred = np.concatenate(np_pred)
    if verbose:
        print(f"Evaluating {np_gt.shape[0]} poses")

    avg_results, results = frame_wise_eval(pred_3d=np_pred, gt_3d=np_gt,
                                    root_index=H36MOrder17P.pelvis)
    if return_metrics_per_frame:
        return results, identifiers
    print_results(avg_results, results, model_name=model_name)


def evaluate_mesh_estimation_model(result_dict, predefined_betas=None, save_path=None, rotation_smplx_model=None, verbose=False, force_recalc=False):
    """
    Regresses joints from mesh estimation results with predefined betas or estimated betas
    :param force_recalc: Normally, this method checks if the computation has already been executed and loads the results from the disk if yes. If this flag is set to True, the computation is forced.
    :param rotation_smplx_model: Rotates the SMPLX model by the given rotation matrix after mesh generation
    :param result_dict: dict with necessary information about pose and shape for SMPLX models
    :param predefined_betas: use given predefined beta parameters for SMPLX model instead of estimated ones in the result dict
    :param save_path: path for the resulting meshes to be saved
    :return:
    """

    if not force_recalc and save_path is not None and os.path.exists(save_path):
        print("Using already calculated and saved results")
        with open(save_path, "rb") as f:
            result_dict = pickle.load(f)
            return result_dict

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    gender = "neutral" if predefined_betas is None else predefined_betas["gender"]
    num_betas = 10 if predefined_betas is None else predefined_betas["num_betas"]
    model = smplx.create(SMPL_MODEL_DIR, model_type="smplx",
                         gender=gender, use_face_contour=False,
                         num_betas=num_betas).to(device)
    regressor = np.load(ASPSET_REGRESSOR_PATH)
    for s in result_dict:
        iterator = tqdm(result_dict[s]) if verbose else result_dict[s]
        for c in iterator:
            for cam in result_dict[s][c]:
                frames = list(result_dict[s][c][cam].keys())
                # if max(frames) != len(frames) and verbose:
                #     print(f"Missing frames in {s}-{c}-{cam}")
                frame_list = []
                for frame in range(1, max(frames) + 1):
                    if frame in result_dict[s][c][cam]:
                        if "betas" not in result_dict[s][c][cam][frame]:
                            continue
                        smplx_info = result_dict[s][c][cam][frame]
                        betas = predefined_betas[s][None] if predefined_betas is not None else torch.from_numpy(smplx_info["betas"]).to(device)
                        global_orient = torch.from_numpy(smplx_info["global_orient"]).to(device) if "global_orient" in smplx_info else model.global_orient
                        human = model(betas=betas, return_verts=True,
                                      body_pose=torch.from_numpy(smplx_info["body_pose"]).to(device).reshape(1, -1),
                                      global_orient=global_orient,
                                      )
                        vertices = human.vertices[0].detach().cpu().numpy()
                        joints = regressor @ vertices
                        if rotation_smplx_model is not None:
                            joints = (rotation_smplx_model @ joints.T).T
                        frame_list.append(joints)
                    else:
                        frame_list.append(np.zeros((17, 3)))
                result_dict[s][c][cam] = frame_list

    if save_path is not None:
        with open(save_path, "wb") as f:
            pickle.dump(result_dict, f)
    return result_dict


def generate_and_evaluate(smplx_path, output_path, aspset_path, predefined_betas=None, model_name=None, force_recalc=True, rotate_smplx_mesh=False, split="test"):
    """
    Generate the 3D joints from SMPLX params saved in the smplx path and evaluate them
    :param aspset_path: Path to ASPset dataset
    :param rotate_smplx_mesh: rotate mesh if set to True, necessary for some IK variants (e.g. ASPset)
    :param force_recalc: if false, the 3D joints are loaded from disk if they exist, otherwise they are always computed
    :param smplx_path: SMPLX params dict for subject, clip, camera and frame
    :param output_path: path to save the 3D joints
    :param predefined_betas: use these betas if not None instead of the specified ones in the dict
    :param model_name: model name for result printing
    :return:
    """
    with open(smplx_path, "rb") as f:
        res = pickle.load(f)
    rot_mat = np.asarray([[1, 0, 0], [0, 0, 1], [0, -1, 0]]).T if rotate_smplx_mesh else None
    joints_dict = evaluate_mesh_estimation_model(res,
                                                 predefined_betas=predefined_betas,
                                                 save_path=output_path,
                                                 force_recalc=force_recalc,
                                                 rotation_smplx_model=rot_mat,
                                                 verbose=True)
    evaluate_results(aspset_path, joints_dict, ignore_not_predicted=True, model_name=model_name, verbose=False, split=split)


def evaluate_results_with_all_predefined_betas(res_file, output_path, aspset_path, predefined_betas_file=None, run_name="", rotate_smplx_mesh=False, force_recalc=True):
    """
    Evaluate results of mesh estimation model with all predefined beta parameters given
    :param force_recalc: if false, the 3D joints are loaded from disk if they exist, otherwise they are always computed
    :param rotate_smplx_mesh: rotate mesh if set to True, necessary for some IK variants
    :param res_file: file with SMPLX param results
    :param predefined_betas_file: file with all predefined betas to evaluate
    :param run_name: Name to include in the results printing
    :param aspset_path: Path to ASPset dataset
    :param output_path: path to save the 3D joints, will be extended with run name and beta param name
    :return:
    """
    if output_path.endswith(".pkl"):
        output_path = output_path[:-4]
    generate_and_evaluate(smplx_path=res_file,
                          output_path=output_path + f"_{run_name}.pkl",
                          aspset_path=aspset_path,
                          predefined_betas=None,
                          model_name=f"{run_name}",
                          rotate_smplx_mesh=rotate_smplx_mesh,
                          force_recalc=force_recalc)
    if predefined_betas_file is not None:
        all_betas = get_all_beta_params(predefined_betas_file)
        for name, predefined_betas in all_betas.items():
            generate_and_evaluate(smplx_path=res_file,
                                  output_path=output_path + f"_{run_name}_{name}.pkl",
                                  aspset_path=aspset_path,
                                  predefined_betas=predefined_betas,
                                  model_name=f"{run_name};{name}",
                                  rotate_smplx_mesh=rotate_smplx_mesh,
                                  force_recalc=force_recalc)

if __name__ == '__main__':
    import argparse

    # Parse command line arguments
    parser = argparse.ArgumentParser(
            description='ASPset evaluation.')
    parser.add_argument('--res', required=True,
                        help="File with ASPset results in correct dict format (see readme)")
    parser.add_argument('--out', required=True,
                        help="Base path for the files with the saved 3d keypoints. The filename will be extended by the run name and the name of the beta parameters, if given.")
    parser.add_argument('--aspset_path', required=True,
                        help="Path to the ASPset ground truth")
    parser.add_argument('--betas', required=False,
                        help="File with predefined beta parameters.",
                        default=None)
    parser.add_argument('--name', required=False,
                        default="",
                        help="Name of the run, included in results file and printed output.")
    parser.add_argument('--rotate_mesh', dest='rotate', action='store_true', help="Rotate SMPLX mesh, needed for IK applied to ASPset")
    parser.add_argument('--recalc', dest='recalc', action='store_true', help="Force recalculation of 3D joints, otherwise the joints will be loaded if a results file already exists.")

    args = parser.parse_args()

    print("                                                       |     MPJPE|    NMPJPE|   PAMPJPE|")
    evaluate_results_with_all_predefined_betas(args.res,
                                               args.out,
                                               args.aspset_path,
                                               predefined_betas_file=args.betas,
                                               run_name=args.name,
                                               rotate_smplx_mesh=args.rotate,
                                               force_recalc=args.recalc)