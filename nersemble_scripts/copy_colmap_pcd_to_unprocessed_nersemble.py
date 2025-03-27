from argparse import ArgumentParser
from tqdm import tqdm
import os
import shutil

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Script parameters")
    parser.add_argument("--timesteps_w_colmap_path", type=str)
    parser.add_argument("--timesteps_wo_colmap_path", type=str)
    args = parser.parse_args()

    timesteps_folders = sorted(os.listdir(args.timesteps_w_colmap_path))

    for timestep_folder in tqdm(timesteps_folders):
        # remove existing empty folder
        shutil.rmtree(os.path.join(args.timesteps_wo_colmap_path, timestep_folder, "colmap"))
        shutil.copytree(os.path.join(args.timesteps_w_colmap_path, timestep_folder, "colmap"), os.path.join(args.timesteps_wo_colmap_path, timestep_folder, "colmap"))