# standard library
import json
import os

import k3d

# external libraries
import numpy as np
import smplx
import torch
import trimesh
from scipy.spatial import ConvexHull
from smplx.joint_names import JOINT_NAMES

from anthro.measurements.length import Height, Length
from anthro.measurements.torch_util import TorchMeasurer
from anthro.measurements.util import (
    device,
    least_far,
    mesh_plane,
    split_to_indices,
    tmerge,
    to_numpy,
    project_onto_normal_plane
)
from config import VERT_SEGM_PATH, SMPL_MODEL_DIR, CIRCUMFERENCE_CONFIG_PATH

"""
this is the vertex mask used for the masked vertex loss
"""
saved_closest_vertices = [np.array([8976, 8650]), np.array([4442, 7218]), np.array([5484, 5941]), np.array([5519, 5939]), np.array([8976, 5518]), np.array([8793, 5519]), np.array([ 355, 5518]), np.array([4823, 5078]), np.array([7559, 7814]), np.array([4442, 4823]), np.array([7218, 7559]), np.array([4713, 4273]), np.array([7449, 7017]), np.array([3958, 3673]), np.array([6706, 6434]), np.array([3673, 5880]), np.array([6434, 8574]), np.array([5780, 5770]), np.array([8474, 8463]), np.array([8846, 5904]), np.array([8635, 8598]), np.array([8846, 5770]), np.array([8635, 8463]), np.array([6193, 8223, 5498, 3432, 6055, 3978, 6069, 8238, 5943, 3434, 6048,
       5516, 3433, 6279, 5943, 6193, 8223, 5498, 3432, 6055, 3518, 6069,
       8223, 5943, 3434, 6195, 3977, 6726, 5943, 3433, 6047, 8224, 3965,
       3432, 6055, 3294, 6069, 6713, 5502, 3284, 6048, 5516, 6279, 5943,
       3433, 6047, 6714, 3965, 3263, 6277, 3516, 6070, 6713, 5502, 3284,
       6048, 3978, 8238, 5494, 3285, 6047, 8224, 5498, 3306, 6277, 3516,
       6193, 6713, 3966, 3284, 6194, 3518, 6725, 5943, 3433, 6195, 8224,
       3965, 3263, 6057, 3294, 6193, 6713, 5502, 3284, 6194, 3978, 6725,
       5943, 3285, 6048, 3977, 6726, 3295]), np.array([7131, 6813, 6644, 3872, 5644, 4050, 8269, 8183, 4071, 6796, 6814,
       6333, 3381, 5645, 5443, 8178, 6645, 5642, 6134, 8341, 8340, 5449,
       3572, 5556, 3371, 7131, 8339, 6644, 3872, 5644, 3390, 6003, 8183,
       3243, 6796, 8342, 6078, 3381, 5645, 3392, 8178, 6645, 5642, 6133,
       8341, 8340, 3382, 3892, 5442, 3370, 7131, 8339, 6142, 3872, 5648,
       4051, 6003, 5487, 3243, 6796, 8338, 6623, 3896, 4069, 4395, 8178,
       6143, 5642, 6133, 6815, 6640, 3897, 5646, 5556, 3370, 8262, 6813,
       6645, 3315, 4068, 4051, 6815, 5531, 3243, 6796, 8338, 6078, 3896,
       4069, 4395, 8267, 8183, 4071, 6134, 6815, 6333, 3897, 5646, 5444,
       3370, 8262, 8340, 6142, 4452, 5648, 3371, 6006, 5449, 5446, 6150,
       6813, 6078, 3896, 4069, 3392, 8179, 6143, 4071, 6150, 8341, 7188,
       3897, 3892, 5444, 8267, 6640, 6142, 3572, 5647, 3371, 6005, 3382,
       5446, 6150, 8338, 6623, 3896, 5644, 4395, 8179, 6143, 4071, 6796,
       8342, 6333, 3381, 5645, 5444, 8178, 6645, 5642, 6815, 3382, 5556]), np.array([8397, 6595, 3492, 5705, 6890, 3542, 6825, 6303, 5596, 5671, 8364,
       3840, 6305, 8409, 4146, 4081, 8397, 6595, 5715, 5703, 8374, 5682,
       6832, 6303, 5949, 5670, 8365, 5714, 8376, 8409, 5680, 4081, 8399,
       6595, 5715, 4080, 8375, 5682, 6832, 8408, 3493, 5708, 8365, 5714,
       8376, 6253, 4146, 4088, 8399, 8409, 5715, 4088, 6305, 3544, 8398,
       8408, 3493, 5670, 8365, 5714, 8376, 6253, 4146, 4088, 8364, 8409,
       5715, 4081, 6305, 5681, 8398, 6595, 3492, 5705, 8365, 3542, 6832,
       6303, 6254, 5671, 4088, 8402, 8409, 3492, 4088, 8375, 5681, 8397,
       6595, 3492, 5703, 6890, 3542, 6825, 6303, 6254, 5671, 8364, 3840,
       6305, 4146]), np.array([2261, 9172,  628, 1969, 3155, 1309, 3040, 2477, 8968, 2136, 1981,
       3029, 1298, 2610, 2088, 1473, 1968, 2261, 2474,  628,  798, 3065,
       1309, 3040, 2199, 8967, 2040, 1981, 3029, 9323, 2610, 2088, 1322,
        804, 3028, 9172, 1298,  798, 3155, 1309, 3040, 2477, 8967, 1869,
       1965, 2257, 9321, 2480, 2088, 1322,  804, 2261, 2474, 1298, 1969,
       3155,  707, 3025, 2196,  625, 1869, 1965, 3029,  706, 2198, 8967,
       1473,  804, 2261, 2091, 9323,  798, 2166,  707, 3025, 2196,  625,
       1869, 1980, 2952, 1309, 2198, 8967, 1472,  804, 2257, 2091, 9323,
        798, 2480,  707, 3041, 2196,  628, 1969, 2952,  706, 3040, 2477,
       8967, 2136, 1968, 2257, 1298, 2610, 1322]), np.array([5956, 5960, 1213, 5964, 5971, 3208, 3201, 3197, 3195, 5958, 5960,
       5961, 1359, 8940, 3208,  564,  375, 3206, 3193, 5969, 5966, 5961,
       1941, 1359,  462, 3198, 3203, 3192, 5956, 5969, 1213, 2150, 5971,
       3208, 1386,  375, 3195, 5958, 5960, 5961, 5971,  462,  462,  564,
       3198, 3195, 3193, 5956, 5960, 1213, 1941, 1359, 3208, 3201, 3197,
       3195, 5969, 1213, 5961, 1359, 3208, 3201,  375, 3206, 3192, 5955,
       5960, 5961, 5971, 3354,  462,  375, 3195, 3206]), np.array([4046, 3425, 3898, 4171, 3258, 3420, 3900, 3900, 3898, 4042, 4040,
       3259, 3947, 4046, 3425, 4012, 4171, 4008, 3951, 4046, 3425, 3412,
       4171, 3258, 3951, 3900, 3898, 3412, 4522, 4040, 3259, 3947, 3424,
       3425, 4012, 4171, 4008, 3951, 3425, 3412, 4171, 4172, 3951, 3424,
       3900, 4522, 4040, 4008, 3420]), np.array([6181, 6022, 6916, 7258, 6646, 6185, 6699, 8131, 6916, 7258, 6173,
       6185, 6792, 6695, 6755, 6915, 6759, 6646, 6648, 6181, 6755, 6787,
       6759, 6646, 6648, 6181, 6787, 6787, 7258, 6646, 6186, 6792, 6181,
       6022, 6788, 7258, 6186, 6648, 6695, 6022, 6787, 6759, 6173, 6185,
       6181, 8131, 6915, 6646, 6186, 6792]), np.array([4207, 4528, 4255, 4241, 4242, 4527, 4529, 4256, 4531, 4237, 4238,
       4206, 4525, 4204, 4241, 4242, 4206, 4527, 4205, 4530, 4237, 4238,
       4527, 4529, 4256, 4204, 4237, 4242, 4207, 4528, 4255, 4241, 4242,
       4206, 4528, 4205, 4531, 4237, 4238, 4206, 4525, 4204, 4237, 4242]), np.array([7265, 6948, 6981, 6986, 7000, 7264, 7266, 6982, 7263, 6985, 6999,
       7265, 6948, 6951, 7261, 7267, 7264, 6981, 6986, 7000, 6999, 6949,
       6950, 6982, 6982, 6985, 7267, 7265, 6948, 6981, 7261, 7267, 7264,
       6981, 7263, 7000, 6999, 6949, 6950, 6951, 6985, 7267]), np.array([3582, 3581, 3596, 4092, 3599, 3591, 4090, 3775, 3997, 4095, 3578,
       3600, 3582, 3595, 3997, 3577, 3590, 3591, 4090, 3775, 3997, 3599,
       3600, 3582, 3595, 4091, 4095, 3578, 3590, 3582, 3581, 3997, 4092,
       3599, 3591, 4090, 3596, 3997, 3578, 3600, 3582, 3581, 4091, 4095,
       3590, 3604]), np.array([6361, 6339, 6836, 6532, 6834, 6352, 6360, 6839, 6745, 6356, 6342,
       6343, 6351, 6339, 6745, 6357, 6834, 6352, 6360, 6836, 6357, 6834,
       6364, 6360, 6839, 6835, 6342, 6356, 6343, 6361, 6339, 6745, 6357,
       6834, 6352, 6360, 6338, 6835, 6342, 6351, 6339, 6745, 6533, 6343]), np.array([3719, 4001, 4099, 3729, 4160, 3732, 4108, 3726, 4160, 4001, 3730,
       4108, 3718, 3741, 4099, 4108, 3788, 3741, 3730, 3726, 3719, 4001,
       3730, 4108, 3718, 3721, 4101, 4108, 3725, 4160, 3741, 3730, 3725]), np.array([6485, 6491, 6493, 6500, 6479, 6490, 6852, 6493, 6749, 6904, 6488,
       6843, 6499, 6500, 6485, 6491, 6493, 6500, 6480, 6488, 6490, 6845,
       6749, 6904, 6485, 6491, 6493, 6500, 6479, 6852, 6493, 6749, 6480,
       6490, 6843, 6499, 6546])]


def circumference(mesh, plane_origin,
                  plane_normal = np.array([0., 1., 0.]),
                  faces = None,
                  debug = False,
                  visualize = False,
                  cached_signs = None,
                  cached_cases = None):
    """
    Calculates the circumference of a plane-mesh intersection

    Parameters
    -------------

    mesh : Trimesh object
        Mesh to be sliced
    plane_origin : (3,) float
    plane_normal : (3,) float
    faces : (f, ) int
        if in debug mode, this is the face segmentation to be used
        otherwise, these are the allowed face indices
    debug : bool
        if True, this function will return the indices of the
        body parts of the cut faces
    visualize : bool
        if True, returns a k3d visualization of lines
    cached_dots: (n, 3) float
        cached projected dots 
    """
    if debug and faces is not None:
        slice_segments, sliced_faces = trimesh.intersections.mesh_plane(mesh, 
            plane_normal=plane_normal, 
            plane_origin=plane_origin, 
            return_faces=True, cached_dots=cached_signs)
    
        return slice_segments, faces[sliced_faces]

    slice_segments = mesh_plane(mesh, 
        plane_normal=plane_normal, 
        plane_origin=plane_origin, 
        faces = faces,
        signs = cached_signs,
        cases = cached_cases)


    unique_points = np.unique(slice_segments.reshape(-1, 3), axis = 0)
    projected_points = project_onto_normal_plane(unique_points, plane_normal)

    convex_hull = ConvexHull(projected_points)
    
    lines = unique_points[convex_hull.simplices]
    length = np.linalg.norm(lines[:,1,:] - lines[:,0,:], axis = 1).sum()

    if visualize:
        return (length, (unique_points, convex_hull.simplices,
                         slice_segments, visualize_plane(plane_origin,
                                                         plane_normal)))
    return length
    
def visualize_plane(origin, normal, extend = 0.25):
    normal = normal / np.linalg.norm(normal)
    R = np.identity(3)
    z = normal[2]
    if z == -1:
        normal = -normal
    V = np.array([[0, 0, normal[0]], # crossproduct vector
                  [0, 0, -normal[1]],
                  [-normal[0], normal[1], 0]])
    R += V + 1/(1 + z) * (V @ V)
    R = R.T
    p = R @ np.array([[1., 0., 0.], [0., 1., 0.]]).T
    p = p.T
    p[:,0] = -p[:,0]
    return np.concatenate((origin + extend * p, origin - extend * p))


def generate_face_segmentation(point_segmentation: dict, faces: np.ndarray, num_vertices = 10475): # 10475 is the number of vertices in smplx
    """
    generates a face segmentation from a vertex segementation
    """
    body_parts = list(point_segmentation.keys())
    points_bodyparts = np.full(num_vertices, -1, dtype=int)
    for index, body_part in enumerate(body_parts):
        indices = np.array(point_segmentation[body_part])
        points_bodyparts[indices] = index
    bodypart_per_face = points_bodyparts[faces]

    # find_most_common(x, y, z) = y == z ? y : x → find_most_common(arr) = arr[arr[1] == arr[2]]
    faces = bodypart_per_face[np.arange(len(bodypart_per_face)), (bodypart_per_face[:,1] == bodypart_per_face[:,2]).astype(int)]
    return faces, points_bodyparts





class Measurer(object):

    """
    Simple class that implements static methods
    """
    @staticmethod
    def calc_circumference(config, indices, verts, joints, faces, visualize = False):
        result = np.zeros(verts.shape[0])
        axis_joints, landmarks, allowed_faces = config
        i1 = JOINT_NAMES.index(axis_joints[1])
        i2 = JOINT_NAMES.index(axis_joints[0])
        i3 = [indices[i] for i in landmarks]
        normal_vector = joints[:,i1] - joints[:,i2]
        normal_vector /= np.linalg.norm(normal_vector,
                                        axis = 1)[:,np.newaxis]
        normal_plane = np.mean(verts[:,i3], axis = 1)
        cached_dots = np.einsum("ijk,ik->ij", verts - normal_plane[:,np.newaxis,:],
                             normal_vector)

        # sign of the dot product is -1, 0, or 1
        # for whether the point is above or below
        # the plane
        signs = np.zeros_like(cached_dots, dtype=np.int8)
        signs[cached_dots < -tmerge] = -1
        signs[cached_dots > tmerge] = 1
        local_faces = faces[allowed_faces]
        signs = signs[:,local_faces]
        sorted_signs = np.sort(signs, axis = 2).astype(np.int8) # sorting reduces the amount of
        # cases

        coded = (sorted_signs[:, :, 0] << 3) + (sorted_signs[:, :, 1] << 2) + (sorted_signs[:, :, 2] << 1)
        # we are only interested in the cases, where we have one vertex
        # on one side and two vertices on the other side:
        # -1 -1 1 and -1 1 1 -> -8 - 4 + 2 = -10 and -8 + 4 + 2 = -2
        case_basic = np.logical_or(coded == -10, coded == -2)
        # or one vertex on the plane, the other two on different sides: 
        # -1 0 1 -> -8 + 2 = -6
        case_vertex = coded == -6
        # or one edge on the plane, and one corner on either side
        # we also only care about one side, so: 0 0 1 -> 0 + 0 + 2
        case_edge = coded == 2
        viz = None

        for i in range(verts.shape[0]):
            res = circumference(verts[i],
                                  plane_origin = normal_plane[i],
                                  plane_normal = normal_vector[i],
                                  faces=local_faces,
                                  visualize=visualize,
                                  debug = False,
                                  cached_signs = signs[i],
                                  cached_cases = (case_basic[i],
                                                  case_vertex[i],
                                                  case_edge[i]))
            if visualize:
                res, viz = res
            result[i] = res


        if visualize:
            return result, viz
        return result


class SMPLXMeasurer(Measurer, TorchMeasurer):
    smplx_indices = {i.lower(): j for i, j in smplx.vertex_ids.vertex_ids['smplx'].items() }
    smplx_indices.update({
        # using indices from https://github.com/DavidBoja/SMPL-Anthropometry
        "headtop": 8976,
        "lheadtemple": 1980,
        "neckadamapple": 8940,
        "lnipple": 3572,
        "rnipple": 8340,
        "shouldertop": 5616,
        "inseampoint": 5601,
        "bellybutton": 5939,
        "backbellybutton": 5941,
        "crotch": 3797,
        "pubicbone": 5949,
        "lwristdown": 4713,
        "rwristdown": 7449,
        "lwristup": 4823,
        "rwristup": 7559,
        "rbicep": 6788, 
        "rforearm": 7266,
        "lshoulder": 4442,
        "rshoulder": 7218, 
        # "llowhip": 4112, 
        "lthigh": 3577,
        "lcalf": 3732,
        "lankle": 5880,

        # self chosen indices
        "supernastrale": 5519,
        "chin": 8793,
        "cervicale": 5484,
        "rknee": 6434,
        "lknee": 3673,
        "lelbow": 4273,
        "relbow": 7017,
        # "rankle": 5880,
        "rankle": 8574,
        "rcalf": 3732,
        "llowhip": 3958,
        "rlowhip": 6706,
        "lball": 5904,
        "rball": 8598,
        "lbicep": 4042,
        "lforearm": 4530,
        "rthigh": 6338
        })

    smplx_indices["heels"] = (smplx_indices["lheel"], 
                              smplx_indices["rheel"])
    smplx_indices["ear"] = (smplx_indices["rear"], 
                              smplx_indices["lear"])

    length_config = {
        "height":                  Height("headtop", "heels"),
        "shoulder width":          Length("lshoulder", "rshoulder"),
        "torso height from back":  Height("cervicale", "backbellybutton"),
        "torso height from front": Length("supernastrale", "bellybutton"),
        "head":                    Height("headtop", "cervicale"),
        "midline neck":            Length("chin", "supernastrale"),
        "lateral neck":            Height("ear", "cervicale"),
    }

    directional_length_config = {
        "hand":         Length("wristup", ("middle", "ring")),
        "arm":          Length("shoulder", "wristup"),
        "forearm":      Length("wristdown", "elbow"),
        "thigh":        Length("lowhip", "knee"),
        "calf":         Length("knee", "ankle"),
        "footwidth":    Length("smalltoe", "bigtoe"),
        "heel to ball": Length("heel", "ball"),
        "heel to toe":  Length("heel", "bigtoe")
    }

    circumference_preconfig = {
        "waist": (["pelvis", "spine3"], ["bellybutton", "backbellybutton"], ["hips", "spine"]),
        "chest": (["pelvis", "spine3"], ["lnipple", "rnipple"], ['spine1', 'spine2']),
        "hip":   (["pelvis", "spine3"], ["pubicbone"], ["hips", "leftUpLeg", "rightUpLeg"]),
        "head":  (["pelvis", "spine3"], ["lheadtemple"], ["head"]),
        "neck":  (["head"  , "spine2"], ["neckadamapple"], ["neck"])
    }

    directional_circumference_preconfig = {
        "arm":     (["shoulder", "elbow"], ["bicep"], ["Arm"]), # technically around the bicep area
        "forearm": (["elbow", "wrist" ], ["forearm"], ["ForeArm"]),
        "thigh":  (["pelvis", "spine3"], ["thigh"], ["UpLeg"]),
        "calf":   (["pelvis", "spine3"], ["calf"], ["Leg"]),
    }

    generate_dir = False # saves whether the directional configurations
    # are already generated

    @staticmethod
    def generate_directions():
        """
        generates the left and right versions of the directional 
        anthropometric measurements
        """
        for i, j in SMPLXMeasurer.directional_length_config.items():
            l, r = j.get_dir(SMPLXMeasurer.smplx_indices)
            SMPLXMeasurer.length_config["left " + i] = l
            SMPLXMeasurer.length_config["right " + i] = r

        for conf in SMPLXMeasurer.length_config.values():
            conf.comp(SMPLXMeasurer.smplx_indices)

        for i, j in SMPLXMeasurer.directional_circumference_preconfig.items():
            SMPLXMeasurer.circumference_preconfig["left " + i] = SMPLXMeasurer.add_direction_circumference(SMPLXMeasurer.smplx_indices, JOINT_NAMES, j, "left_")
            SMPLXMeasurer.circumference_preconfig["right " + i] = SMPLXMeasurer.add_direction_circumference(SMPLXMeasurer.smplx_indices, JOINT_NAMES, j, "right_")


    def find_missing_mirrorpoint(self, name, verts = None):
        """
        tries to find the missing mirror point in a mesh
        """
        if verts is None:
            verts = self.neutral_zero_human.vertices[0].detach().cpu().numpy()
        pos = verts[SMPLXMeasurer.smplx_indices[name] if type(name) is str else
                    name]
        p = np.array([-pos[0], pos[1], pos[2]])
        return least_far(p, verts)


    @staticmethod
    def add_direction(indices, config, dir = "l"):
        """
        recursively adds a direction prefix to the config
        """
        if type(config) is str:
            r = dir + config
            if r in indices or len(indices) == 0:
                return r
            return config
        if type(config) is tuple or type(config) is list:
            return [SMPLXMeasurer.add_direction(indices, i, dir) for i in config]
        return config

    @staticmethod
    def add_direction_circumference(indices, joints, config, dir="left_"):
        """
        recursively adds a direction prefix to the config for
        circumferencecs
        """
        return (SMPLXMeasurer.add_direction(joints, config[0], dir), ) + (SMPLXMeasurer.add_direction(indices, config[1], dir[0]),) + (SMPLXMeasurer.add_direction([], config[2], dir[:-1]), )

    def generate_circumference_config(self,
                                      preconfig = None,
                                      model = None,
                                      vertex_segmentation = None):
        if vertex_segmentation is None:
            vertex_segmentation = self.vertex_segmentation
        if model is None:
            model = self.neutral_model
        if preconfig is None:
            preconfig = SMPLXMeasurer.circumference_preconfig

        body_parts = np.array(list(vertex_segmentation.keys()))
        face_segmentation, vertex_segmentation = generate_face_segmentation(vertex_segmentation, model.faces)
        face_segmentation_list = split_to_indices(face_segmentation)
        offset = 1 if -1 in face_segmentation else 0
        circumference_config = {i: (j[0], j[1], np.concatenate([
                face_segmentation_list[offset + np.argwhere(body_parts == x)[0][0]] for x in j[2]
            ])) for i, j in SMPLXMeasurer.circumference_preconfig.items()}
        return circumference_config, face_segmentation, face_segmentation_list

    def save_circumference_config(self,circumference_config_path =
                                  CIRCUMFERENCE_CONFIG_PATH):
        torch.save(self.circumference_config, circumference_config_path)

    def __init__(self, circumference_config_path = None, 
                 smplx_path = SMPL_MODEL_DIR,
                 vertex_segmentation_path = VERT_SEGM_PATH):
        if not SMPLXMeasurer.generate_dir:
            SMPLXMeasurer.generate_directions()
            SMPLXMeasurer.generate_dir = True
        self.length_config = SMPLXMeasurer.length_config

        self.neutral_model = smplx.create(smplx_path, model_type="smplx",
            gender="neutral", use_face_contour=False,
            num_betas=11)
        self.neutral_zero_human = self.neutral_model(betas=torch.zeros([1,
                            self.neutral_model.num_betas], dtype=torch.float32), return_verts=True)
        self.vertex_segmentation = json.load(open(vertex_segmentation_path))

        if (circumference_config_path is None or not
            os.path.exists(circumference_config_path)) and not os.path.exists(CIRCUMFERENCE_CONFIG_PATH):
            self.circumference_config, _, _ = self.generate_circumference_config()
            if circumference_config_path is not None:
                self.save_circumference_config(circumference_config_path)
        else:
            self.circumference_config = torch.load(CIRCUMFERENCE_CONFIG_PATH)

    def measure(self, mesh, model, visualize = False, to_dict = False):
        """
        This method measures the anthropometric measurements of a mesh
        with no non-zero global orientation and no pose parameters.
        If a mesh with either a non-zero global orientation or pose 
        parameters is provided, a new mesh with 
        the same beta-parameters is generated.

        Parameters
        ---------

        mesh: SMPLXOutput | dict | (torch.Tensor, torch.Tensor) | torch.Tensor
          mesh(es) to be measured:
          can also measure batchwise input

          If the mesh is torch Tensor, it will be seen as beta
          parameters and a mesh will be generated from that

        model: SMPLX
          model to be used 

        visualize: bool
          whether or not visualization data for k3d should be returned

        to_dict: bool
          whether or not to convert the output from np.ndarray to a more
          human readable dictionary by using the
          `measurement_array_to_dict` method

        Returns 
        ---------

        result: (n, 36) or (36,) float
          the measurements of all the meshes.
          This will return a dictionary instead, if `is_dict` is true

        visualizer_data: dict 
          dictionary containing the keypoints needed to visualize the
          measurements
        """
        # converting to the right types:
        verts = joints = None
        if type(mesh) is torch.Tensor:
            mesh = model(betas = mesh, return_verts = True)
        if type(mesh) is tuple:
            verts, joints = mesh
        elif type(mesh) is dict:
            verts, joints = mesh["v"], mesh["keypoints_3d"]
        elif type(mesh) is smplx.utils.SMPLXOutput:
            if (mesh.global_orient != 0).any() or \
                (mesh.body_pose != 0).any() or (mesh.jaw_pose != 0).any():
                mesh = model(betas = mesh.betas, return_verts = True)
            verts, joints = mesh.vertices, mesh.joints
        verts = to_numpy(verts)
        joints = to_numpy(joints)
        if len(verts.shape) < 3:
            verts = verts[np.newaxis,:,:]
            joints = joints[np.newaxis,:,:]

        to_plot = {"points": [], "lengths": [], "circumferences": []}

        configlen = len(self.length_config) + len(self.circumference_config)
        result = np.empty((verts.shape[0], configlen))
        faces = model if type(model) is np.ndarray else model.faces

        # actually calculating the values
        for index, (title, cfg) in enumerate(self.length_config.items()):
            res = cfg.calc(verts, visualize)
            if visualize:
                res, viz = res
                to_plot["lengths"].append(viz)
            result[..., index] = res

        for index, (title, cfg) in enumerate(self.circumference_config.items()):
            res = self.calc_circumference(config=cfg,
                                          indices=self.smplx_indices,
                                          verts=verts, joints=joints,
                                          faces=faces, visualize = visualize)
            if visualize:
                res, viz = res
                to_plot["circumferences"].append(viz)
            result[..., len(self.length_config) + index] = res

        result1 = self.measurement_array_to_dict(result) if to_dict else result
        if visualize:
            for index, (i, j) in enumerate(self.smplx_indices.items()):
                pos = verts[0][np.array(j)]
                if len(pos.shape) > 1:
                    pos = pos.mean(axis = 0)
                to_plot["points"].append(pos)


            return result1, to_plot
        return result1
            

    def measure_torch(self, mesh, model, visualize = False, to_dict = False, unique = True):
        """
        This is the same method, just with pytorch Tensors instead

        If a use case exists, where this method is needed, you might
        want to optimize it before you use it.
        This method is relatively slow compared to the `measure` method.

        Parameters
        -----------

        unique: bool
            makes calculations faster, but makes the function not differentiable anymore
        """
        # converting to the right types:
        verts = joints = None
        if type(mesh) is tuple:
            verts, joints = mesh
        elif type(mesh) is dict:
            verts, joints = mesh["v"], mesh["keypoints_3d"]
        else:
            verts, joints = mesh.vertices, mesh.joints

        to_plot = {"points": [], "lengths": [], "circumferences": []}

        configlen = len(self.length_config) + len(self.circumference_config)
        result = torch.empty((verts.shape[0], configlen))
        faces = torch.tensor((model if type(model) is np.ndarray else model.faces).astype(int), dtype = torch.int32).to(device)

        # actually calculating the values
        for index, (title, cfg) in enumerate(self.length_config.items()):
            res = self.calc_length_torch(cfg, verts, visualize)
            if visualize:
                res, viz = res
                to_plot["lengths"].append(viz)
            result[..., index] = res

        if len(verts.shape) < 3:
            verts = verts.unsqueeze(0)
            joints = joints.unsqueeze(0)

        for index, (title, cfg) in enumerate(self.circumference_config.items()):
            res = self.calc_circumference_torch(config=cfg,
                                          indices=self.smplx_indices,
                                          verts=verts, joints=joints,
                                          faces=faces, visualize = visualize,
                                          unique=unique)
            if visualize:
                res, viz = res
                to_plot["circumferences"].append(viz)
            result[..., len(self.length_config) + index] = res

        result1 = self.measurement_array_to_dict(result) if to_dict else result
        if visualize:
            for index, (i, j) in enumerate(self.smplx_indices.items()):
                pos = verts[0][np.array(j)]
                if len(pos.shape) > 1:
                    pos = pos.mean(axis = 0)
                to_plot["points"].append(pos)
            return result1, to_plot
        return result1


    def measurement_array_to_dict(self, measurement_array):
        """
        converts a list of measurements to a dictionary of measurements
        """
        i = 0
        result = {}
        for key in self.length_config.keys():
            result[key + " length"] = measurement_array[:,i]
            i += 1
        for key in self.circumference_config.keys():
            result[key + " circumference"] = measurement_array[:,i]
            i += 1
        return result


    def visualize_all(self, visualization_data):
        result = {"points": [], "lengths": [], "circumferences": [],
                  "point-labels": [], "length-labels": [],
                  "circumference-labels": [], "circumference-planes": []}
        for index, (i, j) in enumerate(SMPLXMeasurer.smplx_indices.items()):
            pos = visualization_data["points"][index]
            result["points"].append(k3d.points(np.array([pos]), point_size=0.01,
                                      color = 0xffaa00))
            result["point-labels"].append(k3d.label(i, position = pos, color = 0xffaa00, is_html=True))
        for title, (a, b) in zip(self.length_config.keys(), visualization_data["lengths"]):
            result["lengths"].append(k3d.line([a, b]))
            result["length-labels"].append(k3d.label(title, position = (a + b) / 2, color = 0xffaa00, is_html=True))
        for title, (unique_points, simplices, _, plane) in zip(self.circumference_config.keys(), visualization_data["circumferences"]):
            result["circumferences"].append(k3d.lines(unique_points, simplices, indices_type="segment", width=0.01))
            result["circumference-labels"].append(k3d.label(title, position = unique_points[0], color = 0xffaa00, is_html=True))
            result["circumference-planes"].append(k3d.mesh(
                plane, np.array([[0, 1, 2], [0, 2, 3], [0, 3, 1], [3, 2, 1]]),
                color = 0x88ff88, opacity = 0.3))
        return result

    def update_visualization(self, visualization_result, visualization_data):
        for index, i in enumerate(SMPLXMeasurer.smplx_indices.keys()):
            pos = visualization_data["points"][index]
            visualization_result["points"][index].positions = np.array([pos])
            visualization_result["point-labels"][index].positions = pos
        for index, (a, b) in enumerate(visualization_data["lengths"]):
            visualization_result["lengths"][index].vertices = np.array([a, b])
            visualization_result["length-labels"][index].position = (a + b) / 2
        for index, (unique_points, simplices, _, _) in enumerate(
                      visualization_data["circumferences"]):
            visualization_result["circumferences"][index].vertices = unique_points 
            visualization_result["circumferences"][index].indices = simplices
            visualization_result["circumference-labels"][index].position = unique_points[0] 

    def find_closest_vertices(self):
        m, v = self.measure(self.neutral_zero_human, self.neutral_model,
                            visualize= True)
        indices = []
        for i in v["lengths"]:
            indices.append(set())
            for p in i:
                p = torch.tensor(p)
                a, _, _ = least_far(p, self.neutral_zero_human.vertices[0])
                indices[-1].add(a)
        for i in v["circumferences"]:
            indices.append(set())
            for p in i[0]:
                p = torch.tensor(p)
                a, _, _ = least_far(p, self.neutral_zero_human.vertices[0])
                indices[-1].add(a)
        return [np.array([x.item() for x in i]) for i in indices]

smplx_measurer = SMPLXMeasurer()



