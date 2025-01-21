import point_cloud_utils as pcu
import open3d as o3d
import numpy as np
from argparse import ArgumentParser
from tqdm import tqdm
import os
import re
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

def apply_distance_colormap(values, distance_color_map_path):
    # Create a colormap object (RdYlGn)
    cmap = plt.get_cmap("jet")
    cmap.set_over("black")

    # convert from meters to millimeters
    values *= 1000

    # Fixed ranges in milimeters, so we can compare point clouds with each other
    min_distance = 0
    max_distance = 20

    # Normalize the input values between 0 and 1 for colormap scaling
    norm = mcolors.Normalize(vmin=min_distance, vmax=max_distance, clip=False)

    # Generate RGB colors based on the colormap
    rgb_colors = [cmap(norm(val))[:3] for val in values]  # Extract RGB values (ignore alpha channel)

    # Create a plot
    fig, ax = plt.subplots(figsize=(2, 6))
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])

    # Add colorbar with labels in millimeters
    cbar = plt.colorbar(sm, ax=ax, extend='max')
    cbar.set_label("Value (mm)")
    cbar.set_ticks([min_distance, max_distance])

    # Set the ticks to correspond to min and max values
    cbar.set_ticklabels([f'{min_distance:.2f} mm', f'{max_distance:.2f} mm'])

    ax.set_title("Distance to mesh error")
    ax.axis('off')
    plt.savefig(distance_color_map_path, bbox_inches='tight')

    return np.asarray(rgb_colors)

def apply_similarity_colormap(values, similarity_color_map_path):
    # Create a colormap object
    cmap = plt.get_cmap("jet").reversed()

    # Apply abs function, because we don't care if the normals look in same or inverse direction
    values = abs(values)

    # Fixed ranges in milimeters, so we can compare point clouds with each other
    min_similarity = 0
    max_similarity = 1

    # Normalize the input values between 0 and 1 for colormap scaling
    norm = mcolors.Normalize(vmin=min_similarity, vmax=max_similarity, clip=False)

    # Generate RGB colors based on the colormap
    rgb_colors = [cmap(norm(val))[:3] for val in values]  # Extract RGB values (ignore alpha channel)

    # Create a plot
    fig, ax = plt.subplots(figsize=(2, 6))
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])

    # Add colorbar with labels in millimeters
    cbar = plt.colorbar(sm, ax=ax)
    cbar.set_ticks([min_similarity, max_similarity])

    # Set the ticks to correspond to min and max values
    cbar.set_ticklabels([min_similarity, max_similarity])

    ax.set_title("Similarity to normal error")
    ax.axis('off')
    plt.savefig(similarity_color_map_path, bbox_inches='tight')

    return np.asarray(rgb_colors)

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

    return dists, similarities, pcd_vertices

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
        pcd_distance_2_mesh_path = os.path.join(args.meshes_path, "timestep_{}".format(index), "pointcloud_distance_2_mesh_colored.ply")
        distance_color_map_path = os.path.join(args.meshes_path, "timestep_{}".format(index), "distance_color_map.png")
        similarity_color_map_path = os.path.join(args.meshes_path, "timestep_{}".format(index), "similarity_color_map.png")
        pcd_similarity_2_normal_path = os.path.join(args.meshes_path, "timestep_{}".format(index), "pointcloud_similarity_2_normal_colored.ply")
        mesh_path = os.path.join(args.meshes_path, "timestep_{}".format(index), "recon.ply")
        dists, similarities, pcd_vertices = compute_metrics(path_to_mesh=mesh_path, path_to_point_cloud=pcd_path)
        avg_dist = dists.mean()
        avg_similarity = similarities.mean()
        avg_dists.append(avg_dist)
        avg_similarities.append(avg_similarity)

        dist_colors = apply_distance_colormap(dists, distance_color_map_path)
        pcu.save_mesh_vc(pcd_distance_2_mesh_path, pcd_vertices, dist_colors)

        similarity_colors = apply_similarity_colormap(similarities, similarity_color_map_path)
        pcu.save_mesh_vc(pcd_similarity_2_normal_path, pcd_vertices, similarity_colors)


    avg_dists = np.asarray(avg_dists)
    avg_similarities = np.asarray(avg_similarities)

    print("Average shortest distance between GT point cloud and mesh over all timesteps: {}".format(avg_dists.mean()))
    print("Average similarity between GT point cloud and mesh normals over all timesteps: {}".format(avg_similarities.mean()))


