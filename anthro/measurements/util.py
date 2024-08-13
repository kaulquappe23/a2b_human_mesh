"""
A collection of utility functions used to measure the body
"""

from typing import Tuple
import numpy as np
import torch
import trimesh

device = "cuda:0" if torch.cuda.is_available() else "cpu"

tzero = 1e-13
tmerge = 1e-8

# utility functions for calculating the anthropometric measures from the meshes
def to_numpy(tensor):
    """
    converts a given tensor into a numpy array

    Parameters
    ---------
    tensor: (*) *
      Either pytorch Tensor or an numpy ndarray
    """
    return tensor if type(tensor) is np.ndarray else tensor.detach().numpy() if tensor.device.type == "cpu" else tensor.detach().cpu().numpy()


def find_most_equidistant(points: np.ndarray, iter = 12) -> Tuple[np.ndarray, float]:
    """
    finds the most equidistant point of a set of points iteratively 

    Parameters
    ---------
    points: (m, n) float | (b, m, n) float 
      a point cloud of m n-dimensional points, can also calculate in 
      batches with size b
    iter: int
      number of iterations used to find the most equidistant point

    Returns
    ---------
    point: (n,) float
      most equidistant point
    radius: float
      mean distance from every point to the equidistant one
      effectively a radius, if the pointcloud is a hyperball
    """
    x = points.mean(axis = -2)
    for _ in range(iter):
        diff = points - np.expand_dims(x,axis = -2)
        gap = np.linalg.norm(diff, axis = -1)
        r = gap.mean()
        loss = gap - r # how much differs each gap to the mean gap?
    
        J = - diff / gap[..., None] # jacobian matrix of the loss dependant on x
        x = x - np.matmul(np.linalg.pinv(J), loss[...,None])[...,0]
    return x, np.linalg.norm(points - np.expand_dims(x,axis = -2), axis = -1).mean(axis = -1)


def find_max_len(points1, points2):
    """
    finds the largest difference between two sets of points A and B

    Parameters
    ---------

    points1: (m, n)
      list of m n-dimensional points, point cloud A
    points2: (l, n)
      list of l n-dimensional points, point cloud B
    """
    p1 = np.expand_dims(points1.mean(axis = -2), axis = -2) # centroid of set 1
    p2 = np.expand_dims(points2.mean(axis = -2), axis = -2) # centroid of set 2
    ind1 = np.argmin(np.sum((points1 - p1) * (p2 - p1), axis = -1), axis = -1) # lowest dot product to the vector between centroids is the farthest point
    ind2 = np.argmin(np.sum((points2 - p2) * (p1 - p2), axis = -1), axis = -1)
    mp1 = points1[np.arange(len(points1)),ind1,:] if len(points1.shape) > 2 else points1[ind1,:]
    mp2 = points2[np.arange(len(points2)),ind2,:] if len(points2.shape) > 2 else points2[ind2,:]
    return np.linalg.norm(mp1 - mp2, axis=(-1)), ind1, ind2


def sum_dist_closed(points):
    """
    sums all of the distances of points along the list

    Parameters
    ---------

    points: (m, n)
      list of m n-dimensional points, along the sum of the distances
      is supposed to be calculated
    """
    diff_between = points[:-1] - points[1:]
    if type(points) is np.ndarray:
        closing_edge = np.linalg.norm(points[0] - points[-1])
        return np.linalg.norm(diff_between, axis = 1).sum() + closing_edge
    elif type(points) is torch.Tensor:
        closing_edge = torch.norm(points[0] - points[-1])
        return torch.norm(diff_between, dim = 1).sum() + closing_edge
    print("WRONG TYPE USED FOR sum_dist_closed!!!")


def project_onto_normal_plane(points: np.ndarray, normal, invert = False):
    """
    rotate a list of points by normal vector to [0, 0, 1] and then leave 
    out the z coordinate

    Parameters
    ---------

    points: (m, 3) float
      List of m 3d points, that are supposed to be projected
    normal: (3,) float
      Normal vector, used to define the normal plane
    invert: bool
      whether or not to invert the matrix

    Returns
    ---------
    projected_points: (m, 2)
      List of m 2d points, that are projected on the normal plane of normal
    """
    z = normal[2]
    R = np.identity(3)
    if z != -1 and z != 1:
        V = np.array([[0, 0, normal[0]], # crossproduct vector
                      [0, 0, -normal[1]],
                      [-normal[0], normal[1], 0]])
        R += V + 1/(1 + z) * (V @ V)
    if invert:
        R = R.T
    result = (R @ points.T).T[:,:2]
    if z == -1:
        return -result
    return result




def least_far(point, cloud):
    """
    gets the point of a cloud of points, that's the least far away
    """
    if type(cloud) is np.ndarray:
        x = np.linalg.norm(cloud - point, axis = 1)
        i = np.argmin(x)
        return i, cloud[i], x[i]
    elif type(cloud) is torch.Tensor:
        x = torch.norm(cloud - point, dim = 1)
        i = torch.argmin(x)
        return i, cloud[i], x[i]
    print("WRONG TYPE USED FOR least_far!!!")



def split_to_indices(arr: np.ndarray):
    id_sort = np.argsort(arr)
    sorted = arr[id_sort]
    _, id_start = np.unique(sorted, return_index = True)
    return np.split(id_sort, id_start[1:])




def mesh_plane(
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
        point_intersect, valid = trimesh.intersections.plane_lines(
            plane_origin, plane_normal, vertices[edge_thru.T]
        )
        lines = np.column_stack((vertices[vertex_plane[valid]], point_intersect)).reshape(
            (-1, 2, 3)
        )
        return lines

    def handle_on_edge(signs, faces, vertices):
        # case where two vertices are on the plane and one is off
        edges = faces[signs == 0].reshape((-1, 2))
        points = vertices[edges]
        return points

    def handle_basic(signs, faces, vertices):
        # case where one vertex is on one side and two are on the other
        unique_element = signs != signs.sum(axis = 1)[:,np.newaxis]
        edges = np.column_stack(
            (
                faces[unique_element],
                faces[np.roll(unique_element, 1, axis=1)],
                faces[unique_element],
                faces[np.roll(unique_element, 2, axis=1)],
            )
        ).reshape((-1, 2))
        intersections = plane_lines_unsafe(
            plane_origin, plane_normal, vertices[edges.T]
        )
        # since the data has been pre- culled, any invalid intersections at all
        # means the culling was done incorrectly and thus things are broken
        return intersections.reshape((-1, 2, 3))

    # the (m, 2, 3) line segments
    lines = np.vstack(
            [ handle_basic(signs[cases[0]], faces[cases[0]], vertices), 
              handle_on_vertex(signs[cases[1]], faces[cases[1]], vertices),
              handle_on_edge(signs[cases[2]], faces[cases[2]], vertices)
            ]
    )


    return lines


def plane_lines_unsafe(plane_origin, plane_normal, endpoints):
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
    line_dir /= np.linalg.norm(line_dir, axis = 1)[:,np.newaxis] # normalizing

    t = np.dot(plane_normal, (plane_origin - endpoints[0]).T)
    b = np.dot(plane_normal, line_dir.T)

    d = np.divide(t, b)
    intersection = endpoints[0]
    intersection = intersection + np.reshape(d, (-1, 1)) * line_dir

    return intersection

def quick_index(indices, i) -> np.ndarray:
    """
    a helper function to convert different ways to index 
    into a unified numpy array

    mainly used for measurements
    """
    ret = None
    if type(i) is str:
        ret = np.array(indices[i])
    elif (type(i) is tuple or type(i) is list) and type(i[0]) is str:
        ret = np.array([indices[j] for j in i])
    else:
        ret = np.array(i)


    if len(ret.shape) == 0:
       ret = np.expand_dims(ret, axis = 0)

    return ret



def quick_multi_access(arr, indices, i) -> np.ndarray:
    ir = quick_index(indices, i)
    return arr[...,ir]


