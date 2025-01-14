from argparse import ArgumentParser
from tqdm import tqdm
import os

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Testing script parameters")
    parser.add_argument("--start_timestep_index", default=-1, type=int)
    parser.add_argument("--end_timestep_index", default=-1, type=int)
    parser.add_argument("--configs", type=str)
    parser.add_argument("--source_path", type=str)
    parser.add_argument("--model_path", type=str)
    args = parser.parse_args()

    for index in tqdm(range(args.start_timestep_index, args.end_timestep_index+1)):
        os.system("python mesh_extract_tetrahedra.py -s {} -m {} -r 2 --configs {} --timestep_index {}".format(args.source_path, args.model_path, args.configs, index))

