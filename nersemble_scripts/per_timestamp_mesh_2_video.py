import pyvista as pv
import numpy as np
import argparse
import os
import imageio
import re

def natural_sort_key(s):
    return [int(text) if text.isdigit() else text.lower() for text in re.split(r'(\d+)', s)]

def main():
    # Setup argparse for command line input
    parser = argparse.ArgumentParser(description='Generate videos from multiple PLY mesh files across time steps.')
    parser.add_argument('input_folder', type=str, help='Input folder containing timestep directories')
    args = parser.parse_args()

    # Define camera views
    views = {
        "central": np.array([(0.20431703913092927, 0.06167632410642725, 0.6705773209196494),
                             (0.173655203285254, 0.09827889362582884, 0.08799868459761366),
                             (-0.013934653623657773, 0.9978890454980509, 0.06342931738066715)]),
        "side": np.array([(-0.4446141724280363, 0.02220413436369445, 0.1925918672722044),
                          (0.061506472626835576, 0.1142895778986397, -0.08496880629288477),
                          (-0.11043347933953294, 0.9859003012347951, 0.12571810794831872)]),
        "mouth": np.array([(-0.035437082344058575, -0.031873553586936665, 0.6262673755793701),
                           (0.11900690021478212, 0.05757572660973891, 0.06964933212824993),
                           (-0.0513244516327078, 0.9881642881650219, 0.1445584319917641)]),
    }

    # File types to generate videos for
    ply_files = [
        "recon.ply",
        "pointcloud_distance_2_mesh_colored.ply",
        "pointcloud_similarity_2_normal_colored.ply"
    ]

    # Iterate over all timestep directories and generate videos for each ply file
    for view_name, camera_position in views.items():
        for ply_file in ply_files:
            frames = []
            timestep_dirs = sorted([d for d in os.listdir(args.input_folder) if d.startswith('timestep_')], key=natural_sort_key)
            for dir in timestep_dirs:
                mesh_file = os.path.join(args.input_folder, dir, ply_file)
                if os.path.exists(mesh_file):
                    mesh = pv.read(mesh_file)
                    plotter = pv.Plotter(off_screen=True, window_size=(550, 802))
                    if 'RGBA' in mesh.array_names:
                        plotter.add_mesh(mesh, scalars='RGBA', rgba=True, render_points_as_spheres=True, point_size=3, lighting=False)
                    else:
                        plotter.add_mesh(mesh, color='white')
                    plotter.camera_position = camera_position
                    if view_name == 'mouth':
                        plotter.camera.zoom(2.0)
                    img = plotter.screenshot()
                    frames.append(img)
                    plotter.close()

            # Save video to the input folder
            video_path = os.path.join(args.input_folder, f"{view_name}_{ply_file.replace('.ply', '')}_video.mp4")
            with imageio.get_writer(video_path, fps=30, quality=8) as writer:
                for frame in frames:
                    writer.append_data(frame)
            print(f"Video saved to {video_path}")

if __name__ == "__main__":
    main()
