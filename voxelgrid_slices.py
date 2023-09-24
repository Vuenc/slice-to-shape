import scipy
import numpy as np
from matplotlib import pyplot as plt
import trimesh

# Within this file, rectangular 2D slices of a 3D voxel grid are parametrized by
# - a 3D center point
# - two direction vectors and corresponding lengths

# Computes the four corner points of a slice
def slice_corners(center, dir1, dir2, length1, length2):
    assert(np.isclose(np.inner(dir1, dir2), 0)) # directions should be orthogonal
    center, dir1, dir2 = np.array(center), np.array(dir1)/np.linalg.norm(dir1) * length1/2, np.array(dir2)/np.linalg.norm(dir2) * length2/2
    return np.array([center + dir1 + dir2, center + dir1 - dir2, center - dir1 - dir2, center - dir1 + dir2])

# Computes a grid of 3D point coordinates on a slice with a given resolution (pixel_size).
# Outputs a 3xN array. Each row is a floating point coordinate vector of a point on the slice.
def slice_gridpoints(center, dir1, dir2, length1, length2, pixel_size=1, num_steps1=None, num_steps2=None):
    center, dir1, dir2 = np.array(center), np.array(dir1)/np.linalg.norm(dir1) * length1/2, np.array(dir2)/np.linalg.norm(dir2) * length2/2
    
    if num_steps1 is None:
        num_steps1 = int(length1 / pixel_size)
    if num_steps2 is None:
        num_steps2 = int(length2 / pixel_size)
    
    base = center - dir1 - dir2
    
    dir1points = np.linspace([0.,0,0], 2*dir1, num_steps1)
    dir2points = np.linspace([0.,0,0], 2*dir2, num_steps2)
    
    sample_points = np.array([dir1points[:,i] + dir2points[:,i:i+1] for i in range(3)]) + base.reshape(3,1,1)
    return sample_points.reshape(3,-1)

# Extracts a 2D image of a slice from a 3D volume grid
def extract_slice(voxels, center, dir1, dir2, length1, length2, pixel_size=1):
    num_steps1 = length1 // pixel_size
    num_steps2 = length2 // pixel_size
    sample_points = slice_gridpoints(center, dir1, dir2, length1, length2, pixel_size)
    return scipy.ndimage.map_coordinates(voxels, sample_points).reshape(num_steps1, num_steps2)

# Visualization function, computes a mesh that represents a slice and can be displayed in plots.
def compute_slice_mesh(center, dir1, dir2, length1, length2, color):
    slice_mesh = o3d.geometry.TriangleMesh()
    vertices = slice_corners(center, dir1, dir2, length1, length2)
    slice_mesh.vertices = o3d.utility.Vector3dVector(vertices)
    slice_mesh.triangles = o3d.utility.Vector3iVector([[0,1,2],[0,2,3],[2,1,0],[3,2,0]])
    slice_mesh.compute_vertex_normals()
    slice_mesh.paint_uniform_color(color)
    return slice_mesh

# Visualization function, computes a mesh that represents a slice and can be displayed in plots.
def compute_slice_mesh_trimesh(center, dir1, dir2, length1, length2, color: np.ndarray=None) -> trimesh.Trimesh:
    vertices = slice_corners(center, dir1, dir2, length1, length2)
    face_colors = None if color is None else np.repeat(np.array([color]), 4, axis=0)
    slice_mesh = trimesh.Trimesh(vertices, faces=np.array([[0,1,2],[0,2,3],[2,1,0],[3,2,0]]), face_colors=face_colors)
    return slice_mesh