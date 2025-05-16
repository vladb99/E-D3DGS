import numpy as np
import argparse
import os

def main(folder):
    file_path = os.path.join(folder, "poses_bounds.npy")

    if not os.path.exists(file_path):
        print(f"Error: The file {file_path} does not exist.")
        return

    poses_bounds = np.load(file_path)
    matrix = poses_bounds[:, :15].reshape(-1, 3, 5)

    # Extracting according to oder defined by LLFF:
    # R,R,R,T,H
    # R,R,R,T,W
    # R,R,R,T,F,close_bound,far_bound
    # https://github.com/Fyusion/LLFF/tree/master?tab=readme-ov-file#using-your-own-poses-without-running-colmap
    for i in range(poses_bounds.shape[0]):
        print(f"Camera: {i}")
        print("\tPose:")
        print('\t' + np.array2string(matrix[i, :,:4]).replace('\n', '\n\t'))
        print(f"\tHeight:       {matrix[i, 0, 4]}")
        print(f"\tWidth:        {matrix[i, 1, 4]}")
        print(f"\tFocal:        {matrix[i, 2, 4]}")
        print(f"\tClose bound:  {poses_bounds[i, 15]}")
        print(f"\tFar bound:    {poses_bounds[i, 16]}\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Print poses_bounds.npy")
    parser.add_argument("folder", type=str, help="Folder containing the poses_bounds.npy file")

    args = parser.parse_args()

    main(args.folder)
