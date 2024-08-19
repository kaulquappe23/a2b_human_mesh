# -*- coding: utf-8 -*-
"""
Created on 07.06.24

@author: Katja

"""
import copy
import os
import pickle

import numpy as np
import torch
from smplx import build_layer
from smplx.lbs import batch_rodrigues
from tqdm import tqdm

import config
from dataset.action_wise_eval import frame_wise_eval, print_results
from dataset.fit3d.keypoint_order_fit3d import Fit3DOrder
from dataset.fit3d.util.gt_util import load_fit3d_gt
from dataset.fit3d.util.smplx_util import smplx_cfg
from evaluation.general_eval_utils import get_all_beta_params


def evaluate_results(fit3d_path, prediction_dict, ignore_not_predicted=False, return_metrics_per_frame=False, model_name=None, verbose=False, split="val"):

    fit3d_gt = load_fit3d_gt(fit3d_path, split)
    not_found = {}
    np_gt = []
    np_pred = []
    identifiers = []
    for s in fit3d_gt:
        for camera in fit3d_gt[s]:
            for action in fit3d_gt[s][camera]:
                a = fit3d_gt[s][camera][action]
                p = np.asarray(prediction_dict[s][camera][action])
                if ignore_not_predicted:
                    check_not_predicted_frames = np.sum(p, axis=(1, 2)) == 0
                    if np.sum(check_not_predicted_frames) > 0:
                        if s not in not_found:
                            not_found[s] = {}
                        if camera not in not_found[s]:
                            not_found[s][camera] = {}
                        not_found[s][camera][action] = np.where(check_not_predicted_frames)[0]
                if a.shape[0] != p.shape[0]:
                    min_frames = min(a.shape[0], p.shape[0])
                    max_frames = max(a.shape[0], p.shape[0])
                    if verbose:
                        print(
                            f"Different number of frames in GT and prediction for {s} {camera} {action}! Missing {max_frames - min_frames} frames.")
                    a = a[:min_frames]
                    p = p[:min_frames]

                if s in not_found and camera in not_found[s] and action in not_found[s][camera]:
                    frames_not_found = not_found[s][camera][action]
                    use_fames = [i for i in range(a.shape[0]) if i not in frames_not_found]
                else:
                    use_fames = np.arange(a.shape[0])
                ids = [f"{s}_{camera}_{action}_{i}" for i in use_fames]
                identifiers.extend(ids)
                np_gt.append(a[use_fames])
                np_pred.append(p[use_fames])

    np_gt = np.concatenate(np_gt)
    np_pred = np.concatenate(np_pred)
    if verbose:
        print(f"Evaluating {np_gt.shape[0]} poses")

    avg_results, results = frame_wise_eval(pred_3d=np_pred, gt_3d=np_gt,
                                    root_index=0)
    if return_metrics_per_frame:
        return results, identifiers
    print_results(avg_results, results, model_name=model_name)

def evaluate_mesh_estimation_model(result_dict, predefined_betas=None, save_path=None, verbose=False, force_recalc=False):
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
        ext = predefined_betas["name"] if predefined_betas is not None else ""
        with open(save_path.format(ext), "rb") as f:
            result_dict = pickle.load(f)
            return result_dict

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    gender = "neutral" if predefined_betas is None else predefined_betas["gender"]
    num_betas = 10 if predefined_betas is None else predefined_betas["num_betas"]
    cfg = copy.deepcopy(smplx_cfg)
    cfg["smplx"]["betas"]["num"] = num_betas
    cfg["gender"] = gender
    smplx_model = build_layer(config.SMPL_MODEL_DIR, num_betas=num_betas, **cfg)
    smplx_model = smplx_model.to(device)
    for s in result_dict:
        for camera in result_dict[s]:
            iterator = tqdm(result_dict[s][camera]) if verbose else result_dict[s][camera]
            for action in iterator:
                frames = list(result_dict[s][camera][action].keys())
                if max(frames) != len(frames) and verbose:
                    print(f"Missing frames in {s}-{camera}-{action}")
                frame_list = []
                for frame in range(1, max(frames) + 1):
                    if frame in result_dict[s][camera][action]:
                        if "betas" not in result_dict[s][camera][action][frame]:
                            continue
                        smplx_info = result_dict[s][camera][action][frame]
                        betas = predefined_betas[s][None] if predefined_betas is not None else torch.from_numpy(smplx_info["betas"]).to(device)
                        global_orient = torch.from_numpy(smplx_info["global_orient"]).to(device) if "global_orient" in smplx_info else smplx_model.global_orient
                        if not len(betas.shape) == 2:
                            betas = betas[None]
                        if not global_orient.shape[-2] == 3:
                            global_orient = batch_rodrigues(global_orient)
                        body_pose = torch.from_numpy(smplx_info["body_pose"]).to(device)
                        if not body_pose.shape[-2] == 3:
                            body_pose = body_pose.reshape(21, 3)
                            body_pose = batch_rodrigues(body_pose)
                        human = smplx_model(betas=betas, return_verts=True,
                                      body_pose=body_pose[None],
                                      global_orient=global_orient,
                                      )
                        joints = human.joints[0].detach().cpu().numpy()[Fit3DOrder.from_SMPLX_order()]
                        frame_list.append(joints)
                    else:
                        frame_list.append(np.zeros((Fit3DOrder.num_joints, 3)))
                result_dict[s][camera][action] = frame_list

    if save_path is not None:
        with open(save_path, "wb") as f:
            pickle.dump(result_dict, f)
    return result_dict

def generate_and_evaluate(smplx_path, output_path, fit3d_path, predefined_betas=None, model_name=None, force_recalc=True, split="val"):
    """
    Generate the 3D joints from SMPLX params saved in the base path and evaluate them
    :param force_recalc: if false, the 3D joints are loaded from disk if they exist, otherwise they are always computed
    :param smplx_path: SMPLX params dict for subject, clip, camera and frame
    :param predefined_betas: use these betas if not None instead of the specified ones in the dict
    :param model_name: model name for result printing
    :return:
    """
    with open(smplx_path, "rb") as f:
        res = pickle.load(f)
    joints_dict = evaluate_mesh_estimation_model(res,
                                                 predefined_betas=predefined_betas,
                                                 save_path=output_path,
                                                 force_recalc=force_recalc,
                                                 verbose=True)
    evaluate_results(fit3d_path, joints_dict, ignore_not_predicted=True, model_name=model_name, verbose=False, split=split)


def evaluate_results_with_all_predefined_betas(res_file, output_path, fit3d_path, predefined_betas_file=None, run_name="", force_recalc=True):
    """
    Evaluate results of mesh estimation model with all predefined beta parameters given
    :param force_recalc: if false, the 3D joints are loaded from disk if they exist, otherwise they are always computed
    :param res_file: file with SMPLX param results
    :param predefined_betas_file: file with all predefined betas to evaluate
    :param run_name: Name to include in the results printing
    :param fit3d_path: Path to fit3d dataset
    :param output_path: path to save the 3D joints, will be extended with run name and beta param name
    :return:
    """
    if output_path.endswith(".pkl"):
        output_path = output_path[:-4]
    if predefined_betas_file is not None:
        print(f"Analyzing {res_file} with betas from {predefined_betas_file}")
    generate_and_evaluate(smplx_path=res_file,
                          output_path=output_path + f"_{run_name}.pkl",
                          fit3d_path=fit3d_path,
                          predefined_betas=None,
                          model_name=f"{run_name}",
                          force_recalc=force_recalc)
    if predefined_betas_file is not None:
        all_betas = get_all_beta_params(predefined_betas_file)
        for name, predefined_betas in all_betas.items():
            generate_and_evaluate(smplx_path=res_file,
                                  output_path=output_path + f"_{run_name}.pkl",
                                  fit3d_path=fit3d_path,
                                  predefined_betas=predefined_betas,
                                  model_name=f"{run_name};{name}",
                                  force_recalc=force_recalc)

if __name__ == '__main__':
    import argparse

    # Parse command line arguments
    parser = argparse.ArgumentParser(
            description='fit3D evaluation.')
    parser.add_argument('--res', required=True,
                        help="File with fit3D results in correct dict format (see readme)")
    parser.add_argument('--out', required=True,
                        help="Base path for the files with the saved 3d keypoints. The filename will be extended by the run name and the name of the beta parameters, if given.")
    parser.add_argument('--fit3d_path', required=True,
                        help="Path to the fit3D ground truth")
    parser.add_argument('--betas', required=False,
                        help="File with predefined beta parameters.",
                        default=None)
    parser.add_argument('--name', required=False,
                        default="",
                        help="Name of the run, included in results file and printed output.")
    parser.add_argument('--recalc', dest='recalc', action='store_true',
                        help="Force recalculation of 3D joints, otherwise the joints will be loaded if a results file already exists.")

    args = parser.parse_args()

    print("                                                       |     MPJPE|    NMPJPE|   PAMPJPE|")
    evaluate_results_with_all_predefined_betas(args.res,
                                               args.out,
                                               args.fit3d_path,
                                               predefined_betas_file=args.betas,
                                               run_name=args.name,
                                               force_recalc=args.recalc)