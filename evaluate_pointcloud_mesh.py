import point_cloud_utils as pcu
import open3d as o3d
import numpy as np
from argparse import ArgumentParser
from tqdm import tqdm
import os
import re

def compute_metrics(path_to_mesh, path_to_point_cloud):
    mesh_vertices, mesh_faces = pcu.load_mesh_vf(path_to_mesh)
    mesh_face_normals = pcu.estimate_mesh_face_normals(mesh_vertices, mesh_faces)

    pcd = o3d.io.read_point_cloud(path_to_point_cloud)
    pcd_vertices = np.asarray(pcd.points).astype("f")
    pcd_normals = np.asarray(pcd.normals)

    # Compute the shortest distance between each point in p and the mesh:
    #   dists is a NumPy array of shape (P,) where dists[i] is the
    #   shortest distnace between the point p[i, :] and the mesh (v, f)
    dists, fid, bc = pcu.closest_points_on_mesh(pcd_vertices, mesh_vertices, mesh_faces)

    similarities = []
    for index, point_normal in enumerate(pcd_normals):
        closest_mesh_face = fid[index]
        face_normal = mesh_face_normals[closest_mesh_face]

        # Normalized Dot Product between -1 and 1
        # 1 means the vectors are pointing in same direction
        # -1 means opposite directions
        # 0 means they are perpendicular
        norm1 = point_normal / np.linalg.norm(point_normal)
        norm2 = face_normal / np.linalg.norm(face_normal)
        similarity = np.dot(norm1, norm2)
        similarities.append(similarity)
    similarities = np.asarray(similarities)

    # print("Average shortest distance between GT point cloud and mesh {}".format(dists.mean()))
    # print("Average similarity between GT point cloud and mesh normals {}".format(similarities.mean()))

    return dists.mean(), similarities.mean()

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Testing script parameters")
    parser.add_argument("--meshes_path", type=str)
    parser.add_argument("--scene_path", type=str)
    parser.add_argument("--start_timestep_index", default=-1, type=int)
    parser.add_argument("--end_timestep_index", default=-1, type=int)
    args = parser.parse_args()

    timestep_dirs = [d for d in os.listdir(os.path.join(args.scene_path, "timesteps")) if
                     os.path.isdir(os.path.join(os.path.join(args.scene_path, "timesteps"), d))]
    sorted_timestep_dirs = sorted(timestep_dirs, key=lambda x: int(re.search(r'\d+', x).group()))

    avg_dists = []
    avg_similarities = []

    for index in tqdm(range(args.start_timestep_index, args.end_timestep_index + 1)):
        pcd_path = os.path.join(args.scene_path, "timesteps", sorted_timestep_dirs[index], "colmap", "pointclouds", "pointcloud_16.pcd")
        mesh_path = os.path.join(args.meshes_path, "timestep_{}".format(index), "recon.ply")
        avg_dist, avg_similarity = compute_metrics(path_to_mesh=mesh_path, path_to_point_cloud=pcd_path)
        avg_dists.append(avg_dist)
        avg_similarities.append(avg_similarity)

    avg_dists = np.asarray(avg_dists)
    avg_similarities = np.asarray(avg_similarities)

    print("Average shortest distance between GT point cloud and mesh over all timesteps: {}".format(avg_dists.mean()))
    print("Average similarity between GT point cloud and mesh normals over all timesteps: {}".format(avg_similarities.mean()))


