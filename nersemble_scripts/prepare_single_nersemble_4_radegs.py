import os
import shutil
import argparse
import json
import struct
from scipy.spatial.transform import Rotation as R
import numpy as np
import collections
import open3d as o3d
import cv2
import colour
from colour.characterisation import matrix_augmented_Cheung2004
from colour.utilities import as_float_array
from elias.util import load_json

### Most of the stuff taken from https://github.com/colmap/colmap/blob/main/scripts/python/read_write_model.py#L285 ###

CAMERA_ID = 1

BaseImage = collections.namedtuple(
    "Image", ["id", "qvec", "tvec", "camera_id", "name", "xys", "point3D_ids"]
)

Camera = collections.namedtuple(
    "Camera", ["id", "model", "width", "height", "params"]
)

CameraModel = collections.namedtuple(
    "CameraModel", ["model_id", "model_name", "num_params"]
)

Point3D = collections.namedtuple(
    "Point3D", ["id", "xyz", "rgb", "error", "image_ids", "point2D_idxs"]
)

CAMERA_MODELS = {
    CameraModel(model_id=0, model_name="SIMPLE_PINHOLE", num_params=3),
    CameraModel(model_id=1, model_name="PINHOLE", num_params=4),
    CameraModel(model_id=2, model_name="SIMPLE_RADIAL", num_params=4),
    CameraModel(model_id=3, model_name="RADIAL", num_params=5),
    CameraModel(model_id=4, model_name="OPENCV", num_params=8),
    CameraModel(model_id=5, model_name="OPENCV_FISHEYE", num_params=8),
    CameraModel(model_id=6, model_name="FULL_OPENCV", num_params=12),
    CameraModel(model_id=7, model_name="FOV", num_params=5),
    CameraModel(model_id=8, model_name="SIMPLE_RADIAL_FISHEYE", num_params=4),
    CameraModel(model_id=9, model_name="RADIAL_FISHEYE", num_params=5),
    CameraModel(model_id=10, model_name="THIN_PRISM_FISHEYE", num_params=12),
}
CAMERA_MODEL_IDS = dict(
    [(camera_model.model_id, camera_model) for camera_model in CAMERA_MODELS]
)
CAMERA_MODEL_NAMES = dict(
    [(camera_model.model_name, camera_model) for camera_model in CAMERA_MODELS]
)

def colour_correction_Cheung2004_precomputed(image: np.ndarray, CCM: np.ndarray) -> np.ndarray:
    terms = CCM.shape[-1]
    RGB = as_float_array(image)
    shape = RGB.shape

    RGB = np.reshape(RGB, (-1, 3))

    RGB_e = matrix_augmented_Cheung2004(RGB, terms)

    return np.reshape(np.transpose(np.dot(CCM, np.transpose(RGB_e))), shape)

def convert_faces_2_colmap(root_folder, output_folder, scene_name, timestep, apply_alpha_mask_and_color_correction):
    # Define paths based on the given root folder and the output folder
    images_src_folder = os.path.join(root_folder, 'sequences', scene_name, 'timesteps', timestep , 'images-2x')
    alpha_mask_src_folder = os.path.join(root_folder, 'sequences', scene_name, 'timesteps', timestep , 'alpha_map')
    camera_params_src = os.path.join(root_folder, "calibration", 'camera_params.json')
    pointcloud_src = os.path.join(root_folder, 'sequences', scene_name, 'timesteps', timestep , 'colmap/pointclouds' ,'pointcloud_16.pcd')

    # Define new folder paths
    images_dest_folder = os.path.join(output_folder, 'images')
    sparse_dest_folder = os.path.join(output_folder, 'sparse', '0')

    # Define new file paths
    cameras_bin_file = os.path.join(output_folder, sparse_dest_folder, 'cameras.bin')
    images_bin_file = os.path.join(output_folder, sparse_dest_folder, 'images.bin')
    points3D_bin_file = os.path.join(output_folder, sparse_dest_folder, 'points3D.bin')

    # Create the required directories if they don’t exist
    os.makedirs(images_dest_folder, exist_ok=True)
    os.makedirs(sparse_dest_folder, exist_ok=True)

    color_correction = load_json("scripts/ccm_443.json")

    # Copy images from the source images folder to the destination images folder
    if os.path.exists(images_src_folder) and os.path.isdir(images_src_folder):
        for filename in os.listdir(images_src_folder):
            # e.g. cam_220700191.jpg
            cam_name = filename.split("_")[1].split(".")[0]
            src_file = os.path.join(images_src_folder, filename)
            mask_file = os.path.join(alpha_mask_src_folder, os.path.splitext(filename)[0] + '.png')
            dest_file = os.path.join(images_dest_folder, filename)
            if os.path.isfile(src_file):
                if (apply_alpha_mask_and_color_correction and os.path.exists(alpha_mask_src_folder)): #TODO: add param
                    write_image_masked_and_color_corrected(src_file, mask_file, dest_file, color_correction, cam_name)
                else:
                    shutil.copy2(src_file, dest_file)
        print(f"Copied images to: {images_dest_folder}")
    else:
        print(f"Error: The folder {images_src_folder} does not exist or is not a directory.")

    # Write extrinsics to binary
    images = prepare_extrinsics(camera_params_src)
    write_images_binary(images, images_bin_file)
    print(f"Wrote images binary file to {images_bin_file} successfully.")
    # Debug
    # test_images = read_images_binary(images_bin_file)
    # print(test_images)

    # Write intrinsics to binary
    cameras = prepare_intrinsics(camera_params_src)
    write_cameras_binary(cameras, cameras_bin_file)
    print(f"Wrote cameras binary file to {cameras_bin_file} successfully.")
    # Debug
    # test_cameras = read_cameras_binary(cameras_bin_file)
    # print(test_cameras)

    # Write points3D to binary
    points3D = prepare_points3D(pointcloud_src)
    write_points3D_binary(points3D, points3D_bin_file)
    print(f"Wrote points3D binary file to {points3D_bin_file} successfully.")
    # Debug
    # test_points3D = read_points3D_binary(points3D_bin_file)
    # print(test_points3D)

    # Copy the camera parameters file to the output folder
    if not os.path.exists(camera_params_src):
        print(f"Error: The file {camera_params_src} does not exist.")

    # Copy the point cloud file to the sparse/0 folder
    if not os.path.exists(pointcloud_src):
        print(f"Error: The file {pointcloud_src} does not exist.")

def prepare_points3D(pointcloud_src):
    pcd = o3d.io.read_point_cloud(pointcloud_src)
    points = np.asarray(pcd.points)
    colors = np.asarray(pcd.colors)

    points3D = {}

    for index, (point, color) in enumerate(zip(points, colors)):
        color_255 = (color * 255).astype(np.uint8)

        point3D = Point3D(
            id=index+1,
            xyz=point,
            rgb=color_255,
            error=0,
            image_ids=np.empty([0], dtype=np.uint8),
            point2D_idxs=np.empty([0], dtype=np.int8),
        )

        points3D[point3D.id] = point3D

    return points3D

def prepare_intrinsics(camera_params_path):
    with open(camera_params_path, 'r') as f:
        data = json.load(f)

    intrinsics = data["intrinsics"]
    fx = intrinsics[0][0]
    fy = intrinsics[1][1]
    cx = intrinsics[0][2]
    cy = intrinsics[1][2]

    width = 1100
    height = 1604

    cameras = {}

    camera = Camera(
        id=CAMERA_ID,
        model="PINHOLE",
        width=width,
        height=height,
        params=[fx, fy, cx, cy],
    )

    cameras[CAMERA_ID] = camera

    return cameras

def prepare_extrinsics(camera_params_path):
    with open(camera_params_path, 'r') as f:
        data = json.load(f)

    world_2_cam = data["world_2_cam"]

    images = {}

    idx = 1
    for img_id, matrix in world_2_cam.items():
        matrix = np.array(matrix)
        rotation_matrix = matrix[:3, :3]
        translation_vector = matrix[:3, 3]
        quaternion = R.from_matrix(rotation_matrix).as_quat()
        quaternion = np.asarray([quaternion[3], quaternion[0], quaternion[1], quaternion[2]]) # colman quaternion convention is [w, x, y, z]
        filename = "cam_" + str(img_id) + ".jpg"

        image = BaseImage(
            id=idx,
            qvec=quaternion,
            tvec=translation_vector,
            camera_id=CAMERA_ID,
            name=filename,
            xys=[],
            point3D_ids=[],
        )

        images[idx] = image

        idx += 1

    return images

def write_next_bytes(fid, data, format_char_sequence, endian_character="<"):
    """pack and write to a binary file.
    :param fid:
    :param data: data to send, if multiple elements are sent at the same time,
    they should be encapsuled either in a list or a tuple
    :param format_char_sequence: List of {c, e, f, d, h, H, i, I, l, L, q, Q}.
    should be the same length as the data list or tuple
    :param endian_character: Any of {@, =, <, >, !}
    """
    if isinstance(data, (list, tuple)):
        bytes = struct.pack(endian_character + format_char_sequence, *data)
    else:
        bytes = struct.pack(endian_character + format_char_sequence, data)
    fid.write(bytes)

def write_images_binary(images, path_to_model_file):
    """
    see: src/colmap/scene/reconstruction.cc
        void Reconstruction::ReadImagesBinary(const std::string& path)
        void Reconstruction::WriteImagesBinary(const std::string& path)
    """
    with open(path_to_model_file, "wb") as fid:
        write_next_bytes(fid, len(images), "Q")
        for _, img in images.items():
            write_next_bytes(fid, img.id, "i")
            write_next_bytes(fid, img.qvec.tolist(), "dddd")
            write_next_bytes(fid, img.tvec.tolist(), "ddd")
            write_next_bytes(fid, img.camera_id, "i")
            for char in img.name:
                write_next_bytes(fid, char.encode("utf-8"), "c")
            write_next_bytes(fid, b"\x00", "c")
            write_next_bytes(fid, len(img.point3D_ids), "Q")
            for xy, p3d_id in zip(img.xys, img.point3D_ids):
                write_next_bytes(fid, [*xy, p3d_id], "ddq")

def write_cameras_binary(cameras, path_to_model_file):
    """
    see: src/colmap/scene/reconstruction.cc
        void Reconstruction::WriteCamerasBinary(const std::string& path)
        void Reconstruction::ReadCamerasBinary(const std::string& path)
    """
    with open(path_to_model_file, "wb") as fid:
        write_next_bytes(fid, len(cameras), "Q")
        for _, cam in cameras.items():
            model_id = CAMERA_MODEL_NAMES[cam.model].model_id
            camera_properties = [cam.id, model_id, cam.width, cam.height]
            write_next_bytes(fid, camera_properties, "iiQQ")
            for p in cam.params:
                write_next_bytes(fid, float(p), "d")
    return cameras

def write_points3D_binary(points3D, path_to_model_file):
    """
    see: src/colmap/scene/reconstruction.cc
        void Reconstruction::ReadPoints3DBinary(const std::string& path)
        void Reconstruction::WritePoints3DBinary(const std::string& path)
    """
    with open(path_to_model_file, "wb") as fid:
        write_next_bytes(fid, len(points3D), "Q")
        for _, pt in points3D.items():
            write_next_bytes(fid, pt.id, "Q")
            write_next_bytes(fid, pt.xyz.tolist(), "ddd")
            write_next_bytes(fid, pt.rgb.tolist(), "BBB")
            write_next_bytes(fid, pt.error, "d")
            track_length = pt.image_ids.shape[0]
            write_next_bytes(fid, track_length, "Q")
            for image_id, point2D_id in zip(pt.image_ids, pt.point2D_idxs):
                write_next_bytes(fid, [image_id, point2D_id], "ii")

def read_next_bytes(fid, num_bytes, format_char_sequence, endian_character="<"):
    """Read and unpack the next bytes from a binary file.
    :param fid:
    :param num_bytes: Sum of combination of {2, 4, 8}, e.g. 2, 6, 16, 30, etc.
    :param format_char_sequence: List of {c, e, f, d, h, H, i, I, l, L, q, Q}.
    :param endian_character: Any of {@, =, <, >, !}
    :return: Tuple of read and unpacked values.
    """
    data = fid.read(num_bytes)
    return struct.unpack(endian_character + format_char_sequence, data)

def read_images_binary(path_to_model_file):
    """
    see: src/colmap/scene/reconstruction.cc
        void Reconstruction::ReadImagesBinary(const std::string& path)
        void Reconstruction::WriteImagesBinary(const std::string& path)
    """
    images = {}
    with open(path_to_model_file, "rb") as fid:
        num_reg_images = read_next_bytes(fid, 8, "Q")[0]
        for _ in range(num_reg_images):
            binary_image_properties = read_next_bytes(
                fid, num_bytes=64, format_char_sequence="idddddddi"
            )
            image_id = binary_image_properties[0]
            qvec = np.array(binary_image_properties[1:5])
            tvec = np.array(binary_image_properties[5:8])
            camera_id = binary_image_properties[8]
            binary_image_name = b""
            current_char = read_next_bytes(fid, 1, "c")[0]
            while current_char != b"\x00":  # look for the ASCII 0 entry
                binary_image_name += current_char
                current_char = read_next_bytes(fid, 1, "c")[0]
            image_name = binary_image_name.decode("utf-8")
            num_points2D = read_next_bytes(
                fid, num_bytes=8, format_char_sequence="Q"
            )[0]
            x_y_id_s = read_next_bytes(
                fid,
                num_bytes=24 * num_points2D,
                format_char_sequence="ddq" * num_points2D,
            )
            xys = np.column_stack(
                [
                    tuple(map(float, x_y_id_s[0::3])),
                    tuple(map(float, x_y_id_s[1::3])),
                ]
            )
            point3D_ids = np.array(tuple(map(int, x_y_id_s[2::3])))
            images[image_id] = BaseImage(
                id=image_id,
                qvec=qvec,
                tvec=tvec,
                camera_id=camera_id,
                name=image_name,
                xys=xys,
                point3D_ids=point3D_ids,
            )
    return images

def read_cameras_binary(path_to_model_file):
    """
    see: src/colmap/scene/reconstruction.cc
        void Reconstruction::WriteCamerasBinary(const std::string& path)
        void Reconstruction::ReadCamerasBinary(const std::string& path)
    """
    cameras = {}
    with open(path_to_model_file, "rb") as fid:
        num_cameras = read_next_bytes(fid, 8, "Q")[0]
        for _ in range(num_cameras):
            camera_properties = read_next_bytes(
                fid, num_bytes=24, format_char_sequence="iiQQ"
            )
            camera_id = camera_properties[0]
            model_id = camera_properties[1]
            model_name = CAMERA_MODEL_IDS[camera_properties[1]].model_name
            width = camera_properties[2]
            height = camera_properties[3]
            num_params = CAMERA_MODEL_IDS[model_id].num_params
            params = read_next_bytes(
                fid,
                num_bytes=8 * num_params,
                format_char_sequence="d" * num_params,
            )
            cameras[camera_id] = Camera(
                id=camera_id,
                model=model_name,
                width=width,
                height=height,
                params=np.array(params),
            )
        assert len(cameras) == num_cameras
    return cameras

def read_points3D_binary(path_to_model_file):
    """
    see: src/colmap/scene/reconstruction.cc
        void Reconstruction::ReadPoints3DBinary(const std::string& path)
        void Reconstruction::WritePoints3DBinary(const std::string& path)
    """
    points3D = {}
    with open(path_to_model_file, "rb") as fid:
        num_points = read_next_bytes(fid, 8, "Q")[0]
        for _ in range(num_points):
            binary_point_line_properties = read_next_bytes(
                fid, num_bytes=43, format_char_sequence="QdddBBBd"
            )
            point3D_id = binary_point_line_properties[0]
            xyz = np.array(binary_point_line_properties[1:4])
            rgb = np.array(binary_point_line_properties[4:7])
            error = np.array(binary_point_line_properties[7])
            track_length = read_next_bytes(
                fid, num_bytes=8, format_char_sequence="Q"
            )[0]
            track_elems = read_next_bytes(
                fid,
                num_bytes=8 * track_length,
                format_char_sequence="ii" * track_length,
            )
            image_ids = np.array(tuple(map(int, track_elems[0::2])))
            point2D_idxs = np.array(tuple(map(int, track_elems[1::2])))
            points3D[point3D_id] = Point3D(
                id=point3D_id,
                xyz=xyz,
                rgb=rgb,
                error=error,
                image_ids=image_ids,
                point2D_idxs=point2D_idxs,
            )
    return points3D

def write_image_masked_and_color_corrected(path_to_rgb_image, path_to_alpha_mask, path_to_output_image, color_correction, cam_name):
    """
    Applies a masking to an RGB image using a provided alpha mask, then writes the result to a new file.
    """
    rgb_image = cv2.imread(path_to_rgb_image)
    alpha_mask = cv2.imread(path_to_alpha_mask, cv2.IMREAD_GRAYSCALE)

    # Resize alpha mask to match RGB image
    alpha_mask_resized = cv2.resize(alpha_mask, (rgb_image.shape[1], rgb_image.shape[0]), interpolation=cv2.INTER_AREA)
    
    # Normalize to range 0-1
    normalized_alpha_mask = alpha_mask_resized / 255.0
      
    background_color = np.array([255, 255, 255])
    for i in range(3):
        rgb_image[:,:,i] = rgb_image[:,:,i] * normalized_alpha_mask + background_color[i] * (1 - normalized_alpha_mask)

    ccm = np.array(color_correction[cam_name])
    # Apply color correction to image
    image_linear = colour.cctf_decoding(rgb_image / 255.)
    image_corrected = colour_correction_Cheung2004_precomputed(image_linear, ccm)
    image_corrected = np.clip(colour.cctf_encoding(image_corrected) * 255, 0, 255).astype(np.uint8)

    cv2.imwrite(path_to_output_image, image_corrected)

if __name__ == '__main__':
    # Set up argument parsing
    parser = argparse.ArgumentParser(description="Organize files into a new folder structure.")
    parser.add_argument("root_folder", type=str,
                        help="Path to the root folder containing data of the recorded face")
    parser.add_argument("scene_name", type=str,
                        help="Name of the scene")
    parser.add_argument("output_folder", type=str,
                        help="Path to the output folder where the new structure will be created")
    parser.add_argument("timestep", type=str,
                        help="Name of the timestep")
    parser.add_argument("--apply_alpha_mask_and_color_correction", action='store_true',
                        help="When set, the alpha mask is used to set masked pixels in the corresponding image to white")

    args = parser.parse_args()

    # Run the organization function with provided arguments
    convert_faces_2_colmap(args.root_folder, args.output_folder, args.scene_name, args.timestep, args.apply_alpha_mask_and_color_correction)
