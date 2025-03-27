import argparse
import os
import cv2
import numpy as np
import json
from dreifus.matrix import Pose, CameraCoordinateConvention, PoseType
import colour
from colour.characterisation import matrix_augmented_Cheung2004
from colour.utilities import as_float_array
from elias.util import load_json
from matplotlib import pyplot as plt

NUM_CAMS = 16

def colour_correction_Cheung2004_precomputed(image: np.ndarray, CCM: np.ndarray) -> np.ndarray:
    terms = CCM.shape[-1]
    RGB = as_float_array(image)
    shape = RGB.shape

    RGB = np.reshape(RGB, (-1, 3))

    RGB_e = matrix_augmented_Cheung2004(RGB, terms)

    return np.reshape(np.transpose(np.dot(CCM, np.transpose(RGB_e))), shape)

def convert_emotion_2_video(root_folder, scene_folder, output_folder, cameras, alpha_mask, number_of_frames):
    path_scene_timesteps_folder = os.path.join(root_folder, "sequences", scene_folder, "timesteps")
    timesteps_folders = sorted(os.listdir(path_scene_timesteps_folder))[:number_of_frames]

    alpha_masks_dir_path = os.path.join(output_folder, "alpha_masks")
    os.makedirs(alpha_masks_dir_path, exist_ok=True)

    tongue_segmentations_path = os.path.join(output_folder, "segmentations")
    os.makedirs(tongue_segmentations_path, exist_ok=True)

    #segmentations_path = "/home/vbratulescu/Downloads/407-tongue-annotations/407/sequences/EXP-6-tongue-1/timesteps"
    segmentations_path = "/home/vbratulescu/git/data/Nersemble_unprocessed/037/sequences/EXP-6-tongue-1/timesteps"


    # Determine the width and height from the first image
    frame = cv2.imread(os.path.join(path_scene_timesteps_folder, timesteps_folders[0], "images-2x", "cam_" + cameras[0] + ".jpg"))
    height, width, channels = frame.shape
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    FPS = 30
    color_correction = load_json("scripts/ccm_443.json")
    tongue_color = [168, 91, 73]
    for i, camera in enumerate(cameras):
        output_video = os.path.join(output_folder, f"cam{str(i).zfill(2)}.mp4")
        cam_path = os.path.join(alpha_masks_dir_path, f"cam{str(i).zfill(2)}")
        cam_path_tongue_segmentations = os.path.join(tongue_segmentations_path, f"cam{str(i).zfill(2)}")
        os.makedirs(cam_path, exist_ok=True)
        os.makedirs(cam_path_tongue_segmentations, exist_ok=True)
        video = cv2.VideoWriter(output_video, fourcc, FPS, (width, height))
        for i, timestep in enumerate(timesteps_folders):
            rgb_image = cv2.imread(os.path.join(path_scene_timesteps_folder, timestep, "images-2x", "cam_" + camera + ".jpg"))
            segmentation_image = cv2.imread(os.path.join(segmentations_path, timestep, "facer_segmentation_masks", "color_segmentation_cam_" + camera + ".png"))
            segmentation_image[segmentation_image != tongue_color] = 0
            segmentation_image[segmentation_image == tongue_color] = 255
            cv2.imwrite(os.path.join(cam_path_tongue_segmentations, f"{str(i).zfill(4)}.png"), segmentation_image)

            ccm = np.array(color_correction[camera])
            # Apply color correction to image
            image_linear = colour.cctf_decoding(rgb_image / 255.)
            image_corrected = colour_correction_Cheung2004_precomputed(image_linear, ccm)
            image_corrected = np.clip(colour.cctf_encoding(image_corrected) * 255, 0, 255).astype(np.uint8)

            if alpha_mask:
                alpha_mask_image = cv2.imread(os.path.join(path_scene_timesteps_folder, timestep, "alpha_map", "cam_" + camera + ".png"), cv2.IMREAD_GRAYSCALE)
                rgb_image, normalized_alpha_mask = apply_alpha_mask(image_corrected, alpha_mask_image)
                cv2.imwrite(os.path.join(cam_path, f"{str(i).zfill(4)}.png"), normalized_alpha_mask)

            video.write(rgb_image)
        video.release()
    cv2.destroyAllWindows()

def create_pose_bounds(root_folder, output_folder):
    camera_params_path = os.path.join(root_folder, "calibration", 'camera_params.json')

    with open(camera_params_path, 'r') as f:
        data = json.load(f)

        # Intrinsics
        intrinsics = data["intrinsics"]
        intrinsics_matrix = np.array(intrinsics)

        # Extriniscs
        world_2_cam = data["world_2_cam"]

        c2w_poses = []
        cameras = []

        for camera_id, matrix in world_2_cam.items():
            matrix = np.array(matrix)
            
            pose = Pose(matrix_or_rotation=matrix[:3, :3], 
                        translation=matrix[:3, 3],
                        camera_coordinate_convention=CameraCoordinateConvention.OPEN_CV,
                        pose_type=PoseType.WORLD_2_CAM)

            pose.change_pose_type(PoseType.CAM_2_WORLD)
            
            # OpenCV (what camera_params.json uses): [right, down, forwards] or [x,-y,-z]
            # poses_bounds.npy: [down, right, backwards] or [-y,x,z]
            mapping = np.array([[0.0, 1.0, 0.0],[1.0, 0.0, 0.0],[0.0, 0.0, -1.0]])            
            
            c2w_poses.append(np.hstack((pose.get_rotation_matrix() @ mapping, (pose.get_translation()).reshape(-1, 1))))
            cameras.append(camera_id)
            
        cameras = np.asarray(cameras)
        c2w_poses = np.asarray(c2w_poses)
        save_poses(output_folder, c2w_poses, intrinsics_matrix)
        
    return cameras

def save_poses(output_folder, c2w_poses, intrinsics_matrix):
    """
    :param output_folder:
    :param poses: Nx3x4 cam_2_world; Camera convention: [down, right, backwards]
    :return:
    """
    # Further information on the format of poses_bounds.npy and how bounds are computed can be found in LLFF repo:
    # https://github.com/Fyusion/LLFF/tree/master?tab=readme-ov-file#using-your-own-poses-without-running-colmap
    # https://github.com/Fyusion/LLFF/blob/c6e27b1ee59cb18f054ccb0f87a90214dbe70482/llff/poses/pose_utils.py#L56-L88
    
    os.makedirs(output_folder, exist_ok=True)
    
    fx = intrinsics_matrix[0][0]
    fy = intrinsics_matrix[1][1]
    cx = intrinsics_matrix[0][2]
    cy = intrinsics_matrix[1][2]

    width = 1100
    height = 1604
    
    print("Warning: Using hardcoded width and height")
    print("Warning: cx and xy are not taken into account")
    print("Warning: Bounds not available. We set them to NaN")

    save_arr = []
    for _, c2w_pose in enumerate(c2w_poses):
        close_depth = np.nan # A realistic value could be 0.5e-3
        inf_depth   = np.nan # A realaisic value could be 2.0e-3
        save_arr.append(np.concatenate([np.hstack((c2w_pose, np.array([height, width, np.mean([fx, fy])]).reshape(-1, 1))).ravel(),
                                        np.array([close_depth,inf_depth])], 0))
    save_arr = np.array(save_arr)

    np.save(os.path.join(output_folder, 'poses_bounds.npy'), save_arr)

def apply_alpha_mask(rgb_image, alpha_mask):
    """
    Applies a masking to an RGB image using a provided alpha mask
    :param rgb_image: cv2 rgb image
    :param alpha_mask: cv2 grayscale image
    :return: cv2 image with alpha mask applied
    """
    # Resize alpha mask to match RGB image
    alpha_mask_resized = cv2.resize(alpha_mask, (rgb_image.shape[1], rgb_image.shape[0]), interpolation=cv2.INTER_AREA)

    # Normalize to range 0-1
    normalized_alpha_mask = alpha_mask_resized / 255.0

    background_color = np.array([255, 255, 255])
    for i in range(3):
        rgb_image[:, :, i] = rgb_image[:, :, i] * normalized_alpha_mask + background_color[i] * (
                    1 - normalized_alpha_mask)

    return rgb_image, alpha_mask_resized

if __name__ == '__main__':
    # Set up argument parsing
    parser = argparse.ArgumentParser(description="Create videos from NerSemble dataset.")
    parser.add_argument("root_folder", type=str,
                        help="Path to the root folder containing data of the recorded face")
    parser.add_argument("scene_folder", type=str,
                        help="Name of the scene folder containing data of the recorded emotion")
    parser.add_argument("output_folder", type=str,
                        help="Path to the output folder where the new structure will be created")
    parser.add_argument("--alpha_mask", action='store_true',
                        help="When set, the alpha mask is used to set masked pixels in the corresponding image to white")
    parser.add_argument("--number_of_frames", type=int, default=None,
                        help="The number of frames to be converted")

    args = parser.parse_args()

    # Run the organization function with provided arguments
    cameras = create_pose_bounds(args.root_folder, args.output_folder)
    convert_emotion_2_video(args.root_folder, args.scene_folder, args.output_folder, cameras, args.alpha_mask, args.number_of_frames)
