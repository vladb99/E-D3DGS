# MIT License

# Copyright (c) 2023 OPPO

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import os 
import cv2 
import glob 
import tqdm 
import numpy as np 
import shutil
import pickle
import sys 
import argparse
sys.path.append(".")
from script.thirdparty.my_utils import posetow2c_matrcs, rotmat2qvec
from script.thirdparty.pre_colmap import * 
from script.thirdparty.helper3dg import getcolmapsinglen3d

import pyvista as pv
from dreifus.pyvista import add_coordinate_axes, add_camera_frustum
from dreifus.matrix import Pose, Intrinsics
from dreifus.camera import CameraCoordinateConvention, PoseType
from PIL import Image
from script.thirdparty.my_utils import qvec2rotmat


def extractframes(videopath):
    cam = cv2.VideoCapture(videopath)
    ctr = 0
    sucess = True
    for i in range(300):
        if os.path.exists(os.path.join(videopath.replace(".mp4", ""), str(i) + ".png")):
            ctr += 1
    if ctr == 300 or ctr == 150: # 150 for 04_truck 
        print("already extracted all the frames, skip extracting")
        return
    ctr = 0
    while ctr < 300:
        try:
            _, frame = cam.read()
            savepath = os.path.join(os.path.dirname(videopath), "images", os.path.basename(videopath)[:-4], str(ctr).zfill(4) + ".png")
            if not os.path.exists(os.path.dirname(savepath)):
                os.makedirs(os.path.dirname(savepath))

            cv2.imwrite(savepath, frame)
            ctr += 1 
        except:
            sucess = False
            print("error")
    cam.release()
    return


def preparecolmapdynerf(folder):
    folderlist = glob.glob(folder + "images/cam**/")
    savedir = os.path.join(folder, "colmap")
    if not os.path.exists(savedir):
        os.mkdir(savedir)
    savedir = os.path.join(savedir, "input")
    if not os.path.exists(savedir):
        os.mkdir(savedir)

    for folder in folderlist :
        imagepath = os.path.join(folder, "0000.png")
        imagesavepath = os.path.join(savedir, folder.split("/")[-2] + ".png")

        shutil.copy(imagepath, imagesavepath)


    
def convertdynerftocolmapdb(path):
    originnumpy = os.path.join(path, "poses_bounds.npy")
    video_paths = sorted(glob.glob(os.path.join(path, 'cam*.mp4')))
    projectfolder = os.path.join(path, "colmap")
    #sparsefolder = os.path.join(projectfolder, "sparse/0")
    manualfolder = os.path.join(projectfolder, "manual")

    # if not os.path.exists(sparsefolder):
    #     os.makedirs(sparsefolder)
    if not os.path.exists(manualfolder):
        os.makedirs(manualfolder)

    savetxt = os.path.join(manualfolder, "images.txt")
    savecamera = os.path.join(manualfolder, "cameras.txt")
    savepoints = os.path.join(manualfolder, "points3D.txt")
    imagetxtlist = []
    cameratxtlist = []
    if os.path.exists(os.path.join(projectfolder, "input.db")):
        os.remove(os.path.join(projectfolder, "input.db"))

    db = COLMAPDatabase.connect(os.path.join(projectfolder, "input.db"))

    db.create_tables()


    with open(originnumpy, 'rb') as numpy_file:
        poses_bounds = np.load(numpy_file)
        poses = poses_bounds[:, :15].reshape(-1, 3, 5)
        params_dict = {}

        llffposes = poses.copy().transpose(1,2,0)
        w2c_matriclist = posetow2c_matrcs(llffposes)
        assert (type(w2c_matriclist) == list)


        for i in range(len(poses)):
            cameraname = os.path.basename(video_paths[i])[:-4]#"cam" + str(i).zfill(2)
            m = w2c_matriclist[i]
            colmapR = m[:3, :3]
            T = m[:3, 3]
            
            H, W, focal = poses[i, :, -1]
            
            colmapQ = rotmat2qvec(colmapR)
            # colmapRcheck = qvec2rotmat(colmapQ)

            imageid = str(i+1)
            cameraid = imageid
            pngname = cameraname + ".png"

            params_dict[pngname] = [colmapQ[0], colmapQ[1], colmapQ[2], colmapQ[3], T[0], T[1], T[2], focal, focal]
            
            line =  imageid + " "

            for j in range(4):
                line += str(colmapQ[j]) + " "
            for j in range(3):
                line += str(T[j]) + " "
            line = line  + cameraid + " " + pngname + "\n"
            empltyline = "\n"
            imagetxtlist.append(line)
            imagetxtlist.append(empltyline)

            focolength = focal
            model, width, height, params = i, W, H, np.array((focolength,  focolength, W//2, H//2,))

            camera_id = db.add_camera(1, width, height, params)
            cameraline = str(i+1) + " " + "PINHOLE " + str(width) +  " " + str(height) + " " + str(focolength) + " " + str(focolength) + " " + str(W//2) + " " + str(H//2) + "\n"
            cameratxtlist.append(cameraline)
            
            image_id = db.add_image(pngname, camera_id,  prior_q=np.array((colmapQ[0], colmapQ[1], colmapQ[2], colmapQ[3])), prior_t=np.array((T[0], T[1], T[2])), image_id=i+1)
            db.commit()
        db.close()


    with open(savetxt, "w") as f:
        for line in imagetxtlist :
            f.write(line)
    with open(savecamera, "w") as f:
        for line in cameratxtlist :
            f.write(line)
    with open(savepoints, "w") as f:
        return params_dict


def visualize_setup(videopath, params_dict):
    """
    videopath: path to the folder containing all videos
    params_dict: dict of parameters for each image. The pose is world_2_camera: {png_name: [QW, QX, QY, QZ, TX, TY, TZ, fx, fy]}
    """
    path_images = os.path.join(videopath, "colmap", "input")

    images = dict()
    serials = []
    world_2_cam_poses = dict()  # serial => world_2_cam_pose

    fy = params_dict[list(params_dict.keys())[0]][7]
    fx = params_dict[list(params_dict.keys())[0]][8]
    intrinsics = Intrinsics(fx, fy, 0, 0)

    for png_name, params in params_dict.items():
        serial = png_name
        serials.append(serial)

        image = Image.open(os.path.join(path_images, png_name))
        image = np.array(image)
        images[serial] = image

        world_2_cam_rotation = qvec2rotmat(params[:4])
        world_2_cam_translation = np.array(params[4:7])

        world_2_cam_pose = Pose(matrix_or_rotation=world_2_cam_rotation, translation=world_2_cam_translation,
                                camera_coordinate_convention=CameraCoordinateConvention.OPEN_CV,
                                pose_type=PoseType.WORLD_2_CAM)
        world_2_cam_poses[serial] = world_2_cam_pose

    # Visualize camera poses and images
    p = pv.Plotter()
    # add_coordinate_axes(p, scale=0.1)
    for serial in serials:
        add_camera_frustum(p, world_2_cam_poses[serial], intrinsics, image=images[serial])
    p.show()


if __name__ == "__main__" :
    parser = argparse.ArgumentParser()
 
    parser.add_argument("--videopath", default="", type=str)
    parser.add_argument("--startframe", default=0, type=int)
    parser.add_argument("--endframe", default=300, type=int)

    args = parser.parse_args()
    videopath = args.videopath

    startframe = args.startframe
    endframe = args.endframe


    if startframe >= endframe:
        print("start frame must smaller than end frame")
        quit()
    if startframe < 0 or endframe > 300:
        print("frame must in range 0-300")
        quit()
    if not os.path.exists(videopath):
        print("path not exist")
        quit()
    
    if not videopath.endswith("/"):
        videopath = videopath + "/"
    
    
    
    #### step1
    if not os.path.exists(os.path.join(videopath, "images")):
        os.makedirs(os.path.join(videopath, "images"))
        print("start extracting 300 frames from videos")
        videoslist = sorted(glob.glob(videopath + "*.mp4"))
        for v in tqdm.tqdm(videoslist):
            extractframes(v)
    
    # # ## step2 prepare colmap input 
    print("start preparing colmap image input")
    preparecolmapdynerf(videopath)


    print("start preparing colmap database input")
    # # ## step 3 prepare colmap db file 
    params_dict = convertdynerftocolmapdb(videopath)

    visualize_setup(videopath, params_dict)

    # ## step 4 run colmap, per frame, if error, reinstall opencv-headless 
    getcolmapsinglen3d(videopath)




