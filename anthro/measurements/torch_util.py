"""
A collection of utility functions used to measure the body with
pytorch Tensors instead of numpy arrays. This file is the equivalent of
util.py, but doesn't implement every algorithm from it.
"""

import torch
import numpy as np
from smplx.joint_names import JOINT_NAMES
from scipy.spatial import ConvexHull

from anthro.measurements.length import Height
from anthro.measurements.util import tzero, tmerge, device


def project_onto_normal_plane_torch(points: np.ndarray, normal):
    """
    rotate by normal vector to [0, 0, 1] and then leave out the z coordinate
    """
    z = normal[2]
    R = torch.eye(3)
    if z != -1 and z != 1:
        V = torch.tensor([[0, 0, normal[0]], # crossproduct vector
                      [0, 0, -normal[1]],
                      [-normal[0], normal[1], 0]])
        R += V + 1/(1 + z) * (V @ V)
    result = (R @ points.T).T[:,:2]
    if z == -1:
        return -result
    return result

def plane_lines_unsafe_torch(plane_origin, plane_normal, endpoints):
    """
    Calculate plane-line intersections under the assumption, that the
    lines will be valid.

    This is an unsafe but faster version of the trimesh plane_lines
    implementation.

    Parameters
    ---------
    plane_origin : (3,) float
        Point on plane
    plane_normal : (3,) float
        Plane normal vector
    endpoints : (2, n, 3) float
        Points defining lines to be tested

    Returns
    ---------
    intersections : (m, 3) float
        Cartesian intersection points
    """
    line_dir = endpoints[1] - endpoints[0]
    line_dir = line_dir / torch.norm(line_dir, dim = 1).unsqueeze(1) # normalizing

    t = plane_normal @ (plane_origin - endpoints[0]).T
    b = plane_normal @ line_dir.T

    d = torch.divide(t, b)
    intersection = endpoints[0]
    intersection = intersection + torch.reshape(d, (-1, 1)) * line_dir

    return intersection

def plane_lines_torch(plane_origin, plane_normal, endpoints, line_segments=True):
    """
    Calculate plane-line intersections under the assumption, that the
    lines will be valid.

    This is an unsafe but faster version of the trimesh plane_lines
    implementation.

    Parameters
    ---------
    plane_origin : (3,) float
        Point on plane
    plane_normal : (3,) float
        Plane normal vector
    endpoints : (2, n, 3) float
        Points defining lines to be tested

    Returns
    ---------
    intersections : (m, 3) float
        Cartesian intersection points
    """
    line_dir = endpoints[1] - endpoints[0]
    line_dir = line_dir / line_dir.norm(dim = 1).unsqueeze(1) # normalizing

    t = plane_normal @ (plane_origin - endpoints[0]).T
    b = plane_normal @ line_dir.T

    plane_normal = plane_normal / plane_normal.norm()
    valid = torch.abs(b) > tzero
    if line_segments:
        test = plane_normal @ (plane_origin - endpoints[1]).T
        different_sides = torch.sign(t) != torch.sign(test)
        nonzero = torch.logical_or(torch.abs(t) > tzero, torch.abs(test) > tzero)
        valid = torch.logical_and(valid, different_sides)
        valid = torch.logical_and(valid, nonzero)

    d = torch.divide(t[valid], b[valid])
    intersection = endpoints[0][valid]
    intersection = intersection + torch.reshape(d, (-1, 1)) * line_dir[valid]

    return intersection, valid

def mesh_plane_torch(
    vertices,
    faces,
    plane_normal,
    plane_origin,
    signs,
    cases
):
    """
    Find a the intersections between a mesh and a plane,
    returning a set of line segments on that plane.

    This is a reduced version of the mesh_plane function  from trimesh for
    better performance.

    Parameters
    ---------
    vertices : 
      vertices of the mesh
    faces : (m,) int
      faces of the relevant section.
    plane_normal : (3,) float
      Normal vector of plane to intersect with mesh
    plane_origin : (3,) float
      Point on plane to intersect with mesh
    signs : 
    cases :

    Returns
    ----------
    lines :  (m, 2, 3) float
      List of 3D line segments in space.
    """

    def handle_on_vertex(signs, faces, vertices):
        # case where one vertex is on plane
        # and two are on different sides
        vertex_plane = faces[signs == 0]
        edge_thru = faces[signs != 0].reshape((-1, 2))
        if edge_thru.shape[0] == 0:
            return torch.tensor([]).reshape(0, 2, 3)
        point_intersect, valid = plane_lines_torch(
            plane_origin, plane_normal, vertices[torch.tensor(edge_thru, dtype = torch.int)].swapaxes(0, 1)
        )
        lines = torch.column_stack((vertices[vertex_plane[valid]], point_intersect)).reshape(
            (-1, 2, 3)
        ).to(device)
        return lines

    def handle_on_edge(signs, faces, vertices):
        # case where two vertices are on the plane and one is off
        edges = faces[signs == 0].reshape((-1, 2))
        points = vertices[edges]
        return points

    def handle_basic(signs, faces, vertices):
        # case where one vertex is on one side and two are on the other
        unique_element = signs != signs.sum(axis = 1).unsqueeze(1)
        edges = torch.column_stack(
            (
                faces[unique_element],
                faces[torch.roll(unique_element, 1, dims=1)],
                faces[unique_element],
                faces[torch.roll(unique_element, 2, dims=1)],
            )
        ).reshape((-1, 2))
        intersections = plane_lines_unsafe_torch(
            plane_origin, plane_normal, vertices[edges].swapaxes(0, 1)
        )
        # since the data has been pre- culled, any invalid intersections at all
        # means the culling was done incorrectly and thus things are broken
        return intersections.reshape((-1, 2, 3))

    # the (m, 2, 3) line segments
    a = handle_basic(signs[cases[0]], faces[cases[0]], vertices)
    b = handle_on_vertex(signs[cases[1]], faces[cases[1]], vertices)
    c = handle_on_edge(signs[cases[2]], faces[cases[2]], vertices)
    lines = torch.vstack(
            [ a, 
              b,
              c
            ]
    )


    return lines


class TorchMeasurer(object):
    """
    Pytorch equivalent of the Measurer class
    """


    @staticmethod
    def calc_length_torch(config, verts: np.ndarray, visualize = False):
        """
        calculates lengths

        can work with either single meshes or batches
        """
        points1 = verts[..., config.from_c, :]
        points2 = verts[..., config.to_c, :]
        
        ret = a = b = None

        if type(config) is Height:
            a = points1.mean(axis=-2)
            b = points2.mean(axis=-2)
            ret = a[...,1] - b[...,1]
        else:
            a = points1.mean(axis=-2)
            b = points2.mean(axis=-2)
            ret = torch.norm(a - b, dim=-1)
        if visualize:
            if type(config) is Height:
                b[..., 0] = a[..., 0]
                b[..., 2] = a[..., 2]
            return ret, (a[0], b[0])
        return ret



    @staticmethod
    def calc_circumference_torch(config, indices, verts, joints, faces, visualize = False, unique = True):
        """
        a helper function to calculate the circumference based on
        pytorch instead of numpy
        """
        result = torch.zeros(verts.shape[0]).to(device)
        axis_joints, landmarks, allowed_faces = config
        i1 = JOINT_NAMES.index(axis_joints[1])
        i2 = JOINT_NAMES.index(axis_joints[0])
        i3 = [indices[i] for i in landmarks]
        normal_vector = joints[:,i1] - joints[:,i2]
        normal_vector = normal_vector / torch.norm(normal_vector, dim = 1).unsqueeze(-1)
        normal_plane = torch.mean(verts[:,i3], dim = 1)
        cached_dots = torch.einsum("ijk,ik->ij", verts - normal_plane.unsqueeze(1), normal_vector)

        # sign of the dot product is -1, 0, or 1
        # for whether the point is above or below
        # the plane
        signs = torch.zeros_like(cached_dots,
                                 dtype=torch.int8).to(device)
        signs[cached_dots < -tmerge] = -1
        signs[cached_dots > tmerge] = 1
        local_faces = faces[allowed_faces]
        signs = signs[:,local_faces]
        sorted_signs, indices = torch.sort(signs, dim = 2) # sorting reduces the amount of cases

        coded = (sorted_signs[:, :, 0] << 3) + (sorted_signs[:, :, 1] << 2) + (sorted_signs[:, :, 2] << 1)
        # we are only interested in the cases, where we have one vertex
        # on one side and two vertices on the other side:
        # -1 -1 1 and -1 1 1 -> -8 - 4 + 2 = -10 and -8 + 4 + 2 = -2
        case_basic = torch.logical_or(coded == -10, coded == -2)
        # or one vertex on the plane, the other two on different sides: 
        # -1 0 1 -> -8 + 2 = -6
        case_vertex = coded == -6
        # or one edge on the plane, and one corner on either side
        # we also only care about one side, so: 0 0 1 -> 0 + 0 + 2
        case_edge = coded == 2
        viz = None

        for i in range(verts.shape[0]):
            slice_segments = mesh_plane_torch(verts[i], 
                plane_normal=normal_vector[i], 
                plane_origin=normal_plane[i], 
                faces = local_faces,
                signs = signs[i],
                cases = (case_basic[i],
                          case_vertex[i],
                          case_edge[i]))

            if unique:
                unique_points = torch.unique(slice_segments.reshape(-1, 3),
                                             dim = 0)
            else:
                unique_points = slice_segments.reshape(-1, 3)
            projected_points = project_onto_normal_plane_torch(unique_points, normal_vector[i])


            convex_hull = ConvexHull(projected_points.cpu().detach().numpy())
            
            lines = unique_points[torch.tensor(convex_hull.simplices,
                                               dtype = torch.int)]
            length = torch.norm(lines[:,1,:] - lines[:,0,:], dim = 1).sum()

            res = None
            if visualize:
                res, viz = length, (unique_points, convex_hull.simplices)
            else:
                res = length
            result[i] = res


        if visualize:
            return result, viz
        return result

