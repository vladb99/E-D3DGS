#adopted from https://github.com/autonomousvision/gaussian-opacity-fields/blob/main/extract_mesh.py
import cv2
import torch
from scipy.constants import alpha

from scene import Scene
import os
from os import makedirs
from gaussian_renderer import render, integrate
import random
from tqdm import tqdm
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, get_combined_args, ModelHiddenParams, OptimizationParams
from gaussian_renderer import GaussianModel
import numpy as np
import trimesh
from tetranerf.utils.extension import cpp
from utils.tetmesh import marching_tetrahedra
from utils.mesh_extraction_utils import get_tetra_points


@torch.no_grad()
def evaluage_alpha(points, views, gaussians, pipeline, background, kernel_size, loaded_iter):
    final_alpha = torch.ones((points.shape[0]), dtype=torch.float32, device="cuda")
    with torch.no_grad():
        for _, view in enumerate(tqdm(views, desc="Rendering progress")):
            if type(view.original_image) == type(None):
                view.load_image()  # for lazy loading (to avoid OOM issue)
            ret = integrate(points, view, gaussians, pipeline, background, kernel_size=kernel_size, loaded_iter=loaded_iter, num_down_emb_c=hyperparam.min_embeddings, num_down_emb_f=hyperparam.min_embeddings)
            alpha_integrated = ret["alpha_integrated"]
            final_alpha = torch.min(final_alpha, alpha_integrated)
        alpha = 1 - final_alpha
    return alpha



@torch.no_grad()
def evaluage_cull_alpha(points, views, masks, gaussians, pipeline, background, kernel_size, loaded_iter):
    # final_sdf = torch.zeros((points.shape[0]), dtype=torch.float32, device="cuda")
    final_sdf = torch.ones((points.shape[0]), dtype=torch.float32, device="cuda")
    weight = torch.zeros((points.shape[0]), dtype=torch.int32, device="cuda")
    with torch.no_grad():
        for cam_id, view in enumerate(tqdm(views, desc="Rendering progress")):
            torch.cuda.empty_cache()
            ret = integrate(points, view, gaussians, pipeline, background, kernel_size, loaded_iter=loaded_iter, num_down_emb_c=hyperparam.min_embeddings, num_down_emb_f=hyperparam.min_embeddings)
            alpha_integrated = ret["alpha_integrated"]
            point_coordinate = ret["point_coordinate"]
            point_coordinate[:,0] = (point_coordinate[:,0]*2+1)/(views[cam_id].image_width-1) - 1
            point_coordinate[:,1] = (point_coordinate[:,1]*2+1)/(views[cam_id].image_height-1) - 1
            rendered_mask = ret["render"][7]
            mask = rendered_mask[None]
            if not view.gt_alpha_mask is None:
                mask = mask * view.gt_alpha_mask.to(mask.device)
            if not masks is None:
                mask = mask * masks[cam_id]
            valid_point_prob = torch.nn.functional.grid_sample(mask.type(torch.float32)[None],point_coordinate[None,None],padding_mode='zeros',align_corners=False)
            valid_point_prob = valid_point_prob[0,0,0]
            valid_point = valid_point_prob>0.5
            final_sdf = torch.where(valid_point, torch.min(alpha_integrated,final_sdf), final_sdf)
            weight = torch.where(valid_point, weight+1, weight)
        final_sdf = torch.where(weight>0,0.5-final_sdf,-100)
    return final_sdf

@torch.no_grad()
def marching_tetrahedra_with_binary_search(model_path, name, iteration, views, gaussians: GaussianModel, pipeline, background, kernel_size, meshes_path, timestep: int, loaded_iter):

    # apply deformation, before getting the tetra points
    means3D = gaussians.get_xyz
    scales = gaussians._scaling
    rotations = gaussians._rotation
    opacity = gaussians._opacity
    shs = gaussians.get_features
    time = torch.tensor(timestep).to(means3D.device).repeat(means3D.shape[0], 1)
    means3D_final, scales_deformed, rotations_deformed, opacity_deformed, shs_final, extras = gaussians._deformation(
        means3D,
        scales,
        rotations,
        opacity,
        time,
        None, gaussians,
        None, shs,
        iter=loaded_iter,
        num_down_emb_c=hyperparam.min_embeddings,
        num_down_emb_f=hyperparam.min_embeddings)

    scales_final, _ = gaussians.apply_scaling_n_opacity_with_3D_filter(opacity=opacity_deformed, scales=scales_deformed)

    # generate tetra points here
    points, points_scale = get_tetra_points(rotation_not_activated=rotations_deformed, xyz=means3D_final, scale_after_3D_filter=scales_final)
    cells = cpp.triangulate(points)

    mask = None
    sdf = evaluage_cull_alpha(points, views, mask, gaussians, pipeline, background, kernel_size, loaded_iter)

    torch.cuda.empty_cache()
    # the function marching_tetrahedra costs much memory, so we move it to cpu.
    verts_list, scale_list, faces_list, _ = marching_tetrahedra(points.cpu()[None], cells.cpu().long(), sdf[None].cpu(), points_scale[None].cpu())
    del points
    del points_scale
    del cells
    end_points, end_sdf = verts_list[0]
    end_scales = scale_list[0]
    end_points, end_sdf, end_scales = end_points.cuda(), end_sdf.cuda(), end_scales.cuda()
    
    faces=faces_list[0].cpu().numpy()
    points = (end_points[:, 0, :] + end_points[:, 1, :]) / 2.
        
    left_points = end_points[:, 0, :]
    right_points = end_points[:, 1, :]
    left_sdf = end_sdf[:, 0, :]
    right_sdf = end_sdf[:, 1, :]
    left_scale = end_scales[:, 0, 0]
    right_scale = end_scales[:, 1, 0]
    distance = torch.norm(left_points - right_points, dim=-1)
    scale = left_scale + right_scale
    
    n_binary_steps = 8
    for step in range(n_binary_steps):
        print("binary search in step {}".format(step))
        mid_points = (left_points + right_points) / 2
        mid_sdf = evaluage_cull_alpha(mid_points, views, mask, gaussians, pipeline, background, kernel_size, loaded_iter)
        mid_sdf = mid_sdf.unsqueeze(-1)
        ind_low = ((mid_sdf < 0) & (left_sdf < 0)) | ((mid_sdf > 0) & (left_sdf > 0))

        left_sdf[ind_low] = mid_sdf[ind_low]
        right_sdf[~ind_low] = mid_sdf[~ind_low]
        left_points[ind_low.flatten()] = mid_points[ind_low.flatten()]
        right_points[~ind_low.flatten()] = mid_points[~ind_low.flatten()]
        points = (left_points + right_points) / 2

        
    mesh = trimesh.Trimesh(vertices=points.cpu().numpy(), faces=faces, process=False)
    # filter
    vertice_mask = (distance <= scale).cpu().numpy()
    face_mask = vertice_mask[faces].all(axis=1)
    mesh.update_vertices(vertice_mask)
    mesh.update_faces(face_mask)

    mesh.export(os.path.join(meshes_path, "recon.ply"))


def get_time_steps(cams) -> [float]:
    time_steps = []
    for cam in cams:
        time_steps.append(cam.time)
    return np.unique(np.array(time_steps))

def extract_mesh(dataset : ModelParams, hyperparam: ModelHiddenParams, opt: OptimizationParams, iteration : int, pipeline : PipelineParams, timestep_index: int):
    with torch.no_grad():
        gaussians = GaussianModel(dataset.sh_degree, hyperparam)
        scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False, duration=hyperparam.total_num_frames, loader=dataset.loader, opt=opt)
        
        gaussians.load_ply(os.path.join(dataset.model_path, "point_cloud", f"iteration_{iteration}", "point_cloud.ply"))

        meshes_path = os.path.join(dataset.model_path, "tetrahedra_meshes", "ours_{}".format(scene.loaded_iter), "timestep_{}".format(timestep_index))
        makedirs(meshes_path, exist_ok=True)

        bg_color = [1,1,1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
        kernel_size = dataset.kernel_size

        cams = scene.getTrainCameras()
        # get only train cameras for specific timestep
        timesteps = get_time_steps(cams=cams)
        timestep = timesteps[timestep_index]
        views = []
        for cam in cams:
            if cam.time == timestep:
                if type(cam.original_image) == type(None):
                    cam.load_image()  # for lazy loading (to avoid OOM issue)
                views.append(cam)

        for view in views:
            alpha_mask = cv2.imread(os.path.join(dataset.source_path, "alpha_masks", view.image_name))
            alpha_mask = alpha_mask[:,:,0]
            alpha_mask_resized = cv2.resize(alpha_mask, (view.image_width, view.image_height), interpolation=cv2.INTER_AREA)
            normalized_resized_alpha_mask = alpha_mask_resized / 255.0
            normalized_resized_alpha_mask = torch.from_numpy(normalized_resized_alpha_mask)
            normalized_resized_alpha_mask = normalized_resized_alpha_mask.unsqueeze(0)
            # import matplotlib.pyplot as plt
            # plt.imshow(cv2.cvtColor(alpha_mask_resized, cv2.COLOR_BGR2RGB))
            # plt.show()
            view.gt_alpha_mask = normalized_resized_alpha_mask

        marching_tetrahedra_with_binary_search(dataset.model_path, "test", iteration, views, gaussians, pipeline, background, kernel_size, meshes_path, timestep, scene.loaded_iter)

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    hyperparam = ModelHiddenParams(parser)
    opt = OptimizationParams(parser)
    parser.add_argument("--iteration", default=80000, type=int)
    parser.add_argument("--timestep_index", default=-1, type=int)
    parser.add_argument("--configs", type=str)
    parser.add_argument("--quiet", action="store_true")
    args = get_combined_args(parser)
    print("Rendering " + args.model_path)
    if args.configs:
        import mmcv
        from utils.params_utils import merge_hparams

        config = mmcv.Config.fromfile(args.configs)
        args = merge_hparams(args, config)
    
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    torch.cuda.set_device(torch.device("cuda:0"))
    
    extract_mesh(model.extract(args), hyperparam.extract(args), opt.extract(args), args.iteration, pipeline.extract(args), args.timestep_index)