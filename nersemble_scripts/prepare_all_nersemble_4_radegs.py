from argparse import ArgumentParser
from tqdm import tqdm
import os

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Script parameters")
    parser.add_argument("--root_folder", type=str)
    parser.add_argument("--scene_name", type=str)
    parser.add_argument("--output_folder", type=str)
    parser.add_argument("--apply_alpha_mask", action='store_true')
    args = parser.parse_args()

    timesteps_folders = sorted(os.listdir(os.path.join(args.root_folder, "sequences", args.scene_name, "timesteps")))

    for timestep_folder in tqdm(timesteps_folders):
        if args.apply_alpha_mask:
            os.system("python scripts/prepare_single_nersemble_4_radegs.py {} {} {} {} --apply_alpha_mask".format(args.root_folder, args.scene_name, os.path.join(args.output_folder, timestep_folder), timestep_folder))
        else:
            os.system("python scripts/prepare_single_nersemble_4_radegs.py {} {} {} {}".format(args.root_folder, args.scene_name, os.path.join(args.output_folder, timestep_folder), timestep_folder))