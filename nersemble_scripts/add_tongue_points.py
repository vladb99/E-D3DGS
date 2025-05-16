import open3d as o3d
import torch
import numpy as np
import argparse
import os

def main(file_path):
    # Load the point cloud
    pcd = o3d.io.read_point_cloud(file_path)
    
    # Convert point cloud points to a PyTorch tensor
    points = torch.tensor(np.asarray(pcd.points), dtype=torch.float32)
    
    # Compute the centroid of the point cloud
    centroid = torch.mean(points, dim=0)
    scale = torch.tensor([0.01, 0.01, 0.015])
    #407
    #offset = torch.tensor([0.0, 0.02, 0.065]) #Right, Up, Forward
    #037
    offset = torch.tensor([0.025, -0.06, 0.055])  # Right, Up, Forward
    normals = torch.randn(5000, 3)
    random_points = centroid + offset + normals * scale

    # Convert the random points back to Open3D format
    random_pcd = o3d.geometry.PointCloud()
    random_pcd.points = o3d.utility.Vector3dVector(random_points.numpy())
    random_pcd.normals = o3d.utility.Vector3dVector(normals.numpy())
    random_pcd.paint_uniform_color([1, 0, 0])  # Red color

    # Combine both point clouds
    combined_pcd = pcd + random_pcd

    # Visualize the point cloud along with the random points
    o3d.visualization.draw_geometries([combined_pcd])

    # Ask the user if they are happy with the result
    user_input = input("Are you happy with the result? (Y/N): ")
    if user_input.lower() == 'y':
        # Save the original file with '_original' appended before the extension
        original_path = os.path.splitext(file_path)[0] + "_original" + os.path.splitext(file_path)[1]
        o3d.io.write_point_cloud(original_path, pcd)
        # Save the combined point cloud over the original file
        o3d.io.write_point_cloud(file_path, combined_pcd)
        print(f"Original file saved as {original_path}.")
        print(f"Modified file saved over the original file at {file_path}.")
    else:
        print("No changes made to the original file.")

if __name__ == "__main__":
    # Set up the argument parser
    parser = argparse.ArgumentParser(description="Load a PLY file and visualize with random samples around its centroid using PyTorch.")
    parser.add_argument("file_path", type=str, help="Path to the PLY file to visualize.")
    
    # Parse arguments
    args = parser.parse_args()
    
    # Call the main function with the provided arguments
    main(args.file_path)
