#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import torch
import math
import numpy as np
from typing import NamedTuple

class BasicPointCloud(NamedTuple):
    points : np.array
    colors : np.array
    normals : np.array
    # times: np.array

def apply_rotation(q1, q2):
    """
    Applies a rotation to a quaternion.
    
    Parameters:
    q1 (Tensor): The original quaternion.
    q2 (Tensor): The rotation quaternion to be applied.
    
    Returns:
    Tensor: The resulting quaternion after applying the rotation.
    """
    # Extract components for readability
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2

    # Compute the product of the two quaternions
    w3 = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    x3 = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y3 = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
    z3 = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
    
    # Combine the components into a new quaternion tensor
    q3 = torch.tensor([w3, x3, y3, z3])

    # Normalize the resulting quaternion
    q3_normalized = q3 / torch.norm(q3)
    
    return q3_normalized

def batch_quaternion_multiply(q1, q2):
    """
    Multiply batches of quaternions.
    
    Args:
    - q1 (torch.Tensor): A tensor of shape [N, 4] representing the first batch of quaternions.
    - q2 (torch.Tensor): A tensor of shape [N, 4] representing the second batch of quaternions.
    
    Returns:
    - torch.Tensor: The resulting batch of quaternions after applying the rotation.
    """
    # Calculate the product of each quaternion in the batch
    w = q1[:, 0] * q2[:, 0] - q1[:, 1] * q2[:, 1] - q1[:, 2] * q2[:, 2] - q1[:, 3] * q2[:, 3]
    x = q1[:, 0] * q2[:, 1] + q1[:, 1] * q2[:, 0] + q1[:, 2] * q2[:, 3] - q1[:, 3] * q2[:, 2]
    y = q1[:, 0] * q2[:, 2] - q1[:, 1] * q2[:, 3] + q1[:, 2] * q2[:, 0] + q1[:, 3] * q2[:, 1]
    z = q1[:, 0] * q2[:, 3] + q1[:, 1] * q2[:, 2] - q1[:, 2] * q2[:, 1] + q1[:, 3] * q2[:, 0]

    # Combine into new quaternions
    q3 = torch.stack((w, x, y, z), dim=1)
    
    # Normalize the quaternions
    norm_q3 = q3 / torch.norm(q3, dim=1, keepdim=True)
    
    return norm_q3

def pix2ndc(v, S):
    return (v * 2.0 + 1.0) / S - 1.0

def ndc2pix(v, S):
    return ((v + 1.0) * S - 1.0) * 0.5

def recordpointshelper(model_path, numpoints, iteration, string):
    txtpath = os.path.join(model_path, "exp_log.txt")
    
    with open(txtpath, 'a') as file:
        file.write("iteration at "+ str(iteration) + "\n")
        file.write(string + " pointsnumber " + str(numpoints) + "\n")

def geom_transform_points(points, transf_matrix):
    P, _ = points.shape
    ones = torch.ones(P, 1, dtype=points.dtype, device=points.device)
    points_hom = torch.cat([points, ones], dim=1)
    points_out = torch.matmul(points_hom, transf_matrix.unsqueeze(0))

    denom = points_out[..., 3:] + 0.0000001
    return (points_out[..., :3] / denom).squeeze(dim=0)

def getWorld2View(R, t):
    Rt = np.zeros((4, 4))
    Rt[:3, :3] = R.transpose()
    Rt[:3, 3] = t
    Rt[3, 3] = 1.0
    return np.float32(Rt)

def getWorld2View2(R, t, translate=np.array([.0, .0, .0]), scale=1.0):
    Rt = np.zeros((4, 4))
    Rt[:3, :3] = R.transpose()
    Rt[:3, 3] = t
    Rt[3, 3] = 1.0

    C2W = np.linalg.inv(Rt)
    cam_center = C2W[:3, 3]
    cam_center = (cam_center + translate) * scale
    C2W[:3, 3] = cam_center
    Rt = np.linalg.inv(C2W)
    return np.float32(Rt)

def getProjectionMatrix(znear, zfar, fovX, fovY): #ndc to 
    tanHalfFovY = math.tan((fovY / 2))
    tanHalfFovX = math.tan((fovX / 2))

    top = tanHalfFovY * znear
    bottom = -top
    right = tanHalfFovX * znear
    left = -right

    P = torch.zeros(4, 4)

    z_sign = 1.0

    P[0, 0] = 2.0 * znear / (right - left)
    P[1, 1] = 2.0 * znear / (top - bottom)
    P[0, 2] = (right + left) / (right - left)
    P[1, 2] = (top + bottom) / (top - bottom)
    P[3, 2] = z_sign
#    P[2, 2] = z_sign * zfar / (zfar - znear)
    P[2, 2] = z_sign * (zfar+znear) / (zfar - znear)

    P[2, 3] = -(zfar * znear) / (zfar - znear)
    return P

# https://stackoverflow.com/a/22064917
#GLdouble perspMatrix[16]={2*fx/w,0,0,0,0,2*fy/h,0,0,2*(cx/w)-1,2*(cy/h)-1,-(far+near)/(far-near),-1,0,0,-2*far*near/(far-near),0};
# https://ksimek.github.io/2013/06/03/calibrated_cameras_in_opengl/


# def getProjectionMatrixCV(znear, zfar, fovX, fovY, cx=0.0, cy=0.0):
#     '''
#     cx and cy range is -1 1 which is the ratio of the image size - 0.5 *2!
    
#     '''
#     tanHalfFovY = math.tan(fovY / 2)
#     tanHalfFovX = math.tan(fovX / 2)

#     top = tanHalfFovY * znear
#     bottom = -top
#     right = tanHalfFovX * znear
#     left = -right

#     # Adjust for off-center projection
#     left += cx * znear
#     right += cx * znear
#     top += cy * znear
#     bottom += cy * znear

#     P = torch.zeros(4, 4)

#     z_sign = 1.0

#     P[0, 0] = 2.0 * znear / (right - left)
#     P[1, 1] = 2.0 * znear / (top - bottom)
#     P[0, 2] = (right + left) / (right - left)
#     P[1, 2] = (top + bottom) / (top - bottom)
#     P[3, 2] = z_sign
#     P[2, 2] = z_sign * zfar / (zfar - znear)
#     P[2, 3] = -(zfar * znear) / (zfar - znear)
#     return P

def getProjectionMatrixCV(znear, zfar, fovX, fovY, cx=0.0, cy=0.0):
    '''
    cx and cy range is -0.5 to 0.5  
    we use cx cy range -0.5 * 0.5
    
    '''
    tanHalfFovY = math.tan(fovY / 2)
    tanHalfFovX = math.tan(fovX / 2)

    top = tanHalfFovY * znear
    bottom = -top
    right = tanHalfFovX * znear
    left = -right

    # Adjust for off-center projection
    deltax = (2 * tanHalfFovX * znear) * cx  # 
    deltay = (2 * tanHalfFovY * znear) * cy  #
    
    left += deltax
    right += deltax
    top += deltay
    bottom += deltay


    # left -= deltax
    # right -= deltax
    # top -= deltay
    # bottom -= deltay


    # left = left(1 +cx)* znear
    # right += cx * znear
    # top += cy * znear
    # bottom += cy * znear

    P = torch.zeros(4, 4)

    z_sign = 1.0

    P[0, 0] = 2.0 * znear / (right - left)
    P[1, 1] = 2.0 * znear / (top - bottom)
    P[0, 2] = (right + left) / (right - left)
    P[1, 2] = (top + bottom) / (top - bottom)
    P[3, 2] = z_sign
#    P[2, 2] = z_sign * zfar / (zfar - znear)
    P[2, 2] = z_sign * (zfar+znear) / (zfar - znear)

    P[2, 3] = -(zfar * znear) / (zfar - znear)
    return P



def fov2focal(fov, pixels):
    return pixels / (2 * math.tan(fov / 2))

def focal2fov(focal, pixels):
    return 2*math.atan(pixels/(2*focal))

# From RaDe-GS
# the following functions depths_double_to_points and depth_double_to_normal are adopted from https://github.com/hugoycj/2dgs-gaustudio/blob/main/utils/graphics_utils.py
def depths_double_to_points(view, depthmap1, depthmap2):
    W, H = view.image_width, view.image_height
    fx = W / (2 * math.tan(view.FoVx / 2.))
    fy = H / (2 * math.tan(view.FoVy / 2.))
    intrins_inv = torch.tensor(
        [[1/fx, 0.,-W/(2 * fx)],
        [0., 1/fy, -H/(2 * fy),],
        [0., 0., 1.0]]
    ).float().cuda()
    grid_x, grid_y = torch.meshgrid(torch.arange(W)+0.5, torch.arange(H)+0.5, indexing='xy')
    points = torch.stack([grid_x, grid_y, torch.ones_like(grid_x)], dim=0).reshape(3, -1).float().cuda()
    rays_d = intrins_inv @ points
    points1 = depthmap1.reshape(1,-1) * rays_d
    points2 = depthmap2.reshape(1,-1) * rays_d
    return points1.reshape(3,H,W), points2.reshape(3,H,W)

def point_double_to_normal(view, points1, points2):
    points = torch.stack([points1, points2],dim=0)
    output = torch.zeros_like(points)
    dx = points[...,2:, 1:-1] - points[...,:-2, 1:-1]
    dy = points[...,1:-1, 2:] - points[...,1:-1, :-2]
    normal_map = torch.nn.functional.normalize(torch.cross(dx, dy, dim=1), dim=1)
    output[...,1:-1, 1:-1] = normal_map
    return output

def depth_double_to_normal(view, depth1, depth2):
    points1, points2 = depths_double_to_points(view, depth1, depth2)
    return point_double_to_normal(view, points1, points2)
###