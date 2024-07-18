"""
Code in this package mostly copied from https://github.com/sminchisescu-research/imar_vision_datasets_tools
Licensed under the MIT License
"""

from smplx import build_layer
import torch

import numpy as np
import copy

smplx_cfg = {'ext': 'npz',
             'extra_joint_path': '',
             'folder': 'transfer_data/body_models',
             'gender': 'neutral',
             'joint_regressor_path': '',
             'model_type': 'smplx',
             'num_expression_coeffs': 10,
             'smplx': {'betas': {'create': True, 'num': 10, 'requires_grad': True},
                       'body_pose': {'create': True, 'requires_grad': True, 'type': 'aa'},
                       'expression': {'create': True, 'num': 10, 'requires_grad': True},
                       'global_rot': {'create': True, 'requires_grad': True, 'type': 'aa'},
                       'jaw_pose': {'create': True, 'requires_grad': True, 'type': 'aa'},
                       'left_hand_pose': {'create': True,
                                          'pca': {'flat_hand_mean': False, 'num_comps': 12},
                                          'requires_grad': True,
                                          'type': 'aa'},
                       'leye_pose': {'create': True, 'requires_grad': True, 'type': 'aa'},
                       'reye_pose': {'create': True, 'requires_grad': True, 'type': 'aa'},
                       'right_hand_pose': {'create': True,
                                           'pca': {'flat_hand_mean': False,
                                                   'num_comps': 12},
                                           'requires_grad': True,
                                           'type': 'aa'},
                       'translation': {'create': True, 'requires_grad': True}},
             'use_compressed': False,
             'use_face_contour': True}

class SMPLXHelper:
    def __init__(self, Models_Path=None, device="cpu"):
        self.device = device
        self.Models_Path = Models_Path
        self.cfg = smplx_cfg
        self.smplx_model = build_layer(self.Models_Path, **self.cfg)
        self.smplx_model.to(device)
        self.image_shape = (900, 900)
           
    def get_world_smplx_params(self, smplx_params):
        world_smplx_params = {key: torch.from_numpy(np.array(smplx_params[key]).astype(np.float32)).to(self.device) for key in smplx_params}
        return world_smplx_params
    
    def get_camera_smplx_params(self, smplx_params, cam_params):
        pelvis = self.smplx_model(betas=torch.from_numpy(np.array(smplx_params['betas']).astype(np.float32)).to(self.device)).joints[:, 0, :].cpu().numpy()
        camera_smplx_params = copy.deepcopy(smplx_params)
        camera_smplx_params['global_orient'] = np.matmul(np.array(smplx_params['global_orient']).transpose(0, 1, 3, 2), np.transpose(cam_params['extrinsics']['R'])).transpose(0, 1, 3, 2)
        camera_smplx_params['transl'] = np.matmul(smplx_params['transl'] + pelvis - cam_params['extrinsics']['T'], np.transpose(cam_params['extrinsics']['R'])) - pelvis
        camera_smplx_params = {key: torch.from_numpy(np.array(camera_smplx_params[key]).astype(np.float32)).to(self.device) for key in camera_smplx_params}
        return camera_smplx_params
    
    def get_template_params(self, batch_size=1):
        smplx_params = {}
        smplx_params_all = self.smplx_model()
        for key1 in ['transl', 'global_orient', 'body_pose', 'betas', 'left_hand_pose', 'right_hand_pose', 'jaw_pose', 'expression', 'leye_pose', 'reye_pose']:
            key2 = key1 if key1 in smplx_params_all else 'jaw_pose'
            smplx_params[key1] = np.repeat(smplx_params_all[key2].cpu().detach().numpy(), batch_size, axis=0)
        smplx_params['transl'][:, 2] = 3
        smplx_params['global_orient'][:, :, 1, 1] = -1
        smplx_params['global_orient'][:, :, 2, 2] = -1
        return smplx_params
    
    def get_template(self):
        smplx_posed_data = self.smplx_model()
        smplx_template = {'vertices': smplx_posed_data.vertices[0].cpu().detach().numpy(), 'triangles': self.smplx_model.faces}
        return smplx_template
