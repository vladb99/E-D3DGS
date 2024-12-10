import torch
import math
from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer
from scene.gaussian_model import GaussianModel
from utils.sh_utils import eval_sh
from time import time as get_time

def render(viewpoint_camera, pc: GaussianModel, pipe, bg_color: torch.Tensor, kernel_size, scaling_modifier=1.0, require_coord: bool = True, require_depth: bool = True, override_color = None, cam_no=None, iter=None, train_coarse=False, num_down_emb_c=5, num_down_emb_f=5, disable_filter3D=True):
    """
    Render the scene. 
    
    Background tensor (bg_color) must be on GPU!
    """
 
    # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
    screenspace_points = torch.zeros_like(pc.get_xyz, dtype=pc.get_xyz.dtype, requires_grad=True, device="cuda") + 0
    try:
        screenspace_points.retain_grad()
    except:
        pass

    # Set up rasterization configuration
    
    means3D = pc.get_xyz
    # if cam_type != "PanopticSports":
    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)
    raster_settings = GaussianRasterizationSettings(
        image_height=torch.tensor(viewpoint_camera.image_height).cuda(),
        image_width=torch.tensor(viewpoint_camera.image_width).cuda(),
        tanfovx=torch.tensor(tanfovx).cuda(),
        tanfovy=torch.tensor(tanfovy).cuda(),
        bg=bg_color.cuda(),
        scale_modifier=torch.tensor(scaling_modifier).cuda(),
        viewmatrix=viewpoint_camera.world_view_transform.cuda(),
        projmatrix=viewpoint_camera.full_proj_transform.cuda(),
        sh_degree=torch.tensor(pc.active_sh_degree).cuda(),
        campos=viewpoint_camera.camera_center.cuda(),
        prefiltered=False,
        debug=pipe.debug,
        kernel_size=kernel_size,
        require_coord=require_coord,
        require_depth=require_depth,
    )
    time = torch.tensor(viewpoint_camera.time).to(means3D.device).repeat(means3D.shape[0],1)
  
    # else:
    #     raster_settings = viewpoint_camera['camera']
    #     time=torch.tensor(viewpoint_camera['time']).to(means3D.device).repeat(means3D.shape[0],1)
        

    rasterizer = GaussianRasterizer(raster_settings=raster_settings)

    # means3D = pc.get_xyz
    # add deformation to each points
    # deformation = pc.get_deformation

    
    means2D = screenspace_points
    opacity = pc._opacity
    shs = pc.get_features

    # If precomputed 3d covariance is provided, use it. If not, then it will be computed from
    # scaling / rotation by the rasterizer.
    scales = None
    rotations = None
    cov3D_precomp = None
    if pipe.compute_cov3D_python:
        cov3D_precomp = pc.get_covariance(scaling_modifier)
    else:
        scales = pc._scaling
        rotations = pc._rotation

    means3D_final, scales_final, rotations_final, opacity_final, shs_final, extras = pc._deformation(means3D, scales, 
        rotations, opacity, time, cam_no, pc, None, shs, iter=iter, num_down_emb_c=num_down_emb_c, num_down_emb_f=num_down_emb_f)

    rotations_final = pc.rotation_activation(rotations_final)

    if disable_filter3D:
        scales_final = pc.scaling_activation(scales_final)
        opacity = pc.opacity_activation(opacity_final)
    else:
        scales_final, opacity = pc.apply_scaling_n_opacity_with_3D_filter(opacity=opacity_final, scales=scales_final)
    colors_precomp = None
    if override_color is None:
        if pipe.convert_SHs_python:
            shs_view = pc.get_features.transpose(1, 2).view(-1, 3, (pc.max_sh_degree+1)**2)
            dir_pp = (pc.get_xyz - viewpoint_camera.camera_center.cuda().repeat(pc.get_features.shape[0], 1))
            dir_pp_normalized = dir_pp/dir_pp.norm(dim=1, keepdim=True)
            sh2rgb = eval_sh(pc.active_sh_degree, shs_view, dir_pp_normalized)
            colors_precomp = torch.clamp_min(sh2rgb + 0.5, 0.0)
        else:
            pass
    else:
        colors_precomp = override_color

    # Rasterize visible Gaussians to image, obtain their radii (on screen). 
    # time3 = get_time()
    depth = None
    outputs = rasterizer(
        means3D = means3D_final,
        means2D = means2D,
        shs = shs_final,
        colors_precomp = colors_precomp,
        opacities = opacity,
        scales = scales_final,
        rotations = rotations_final,
        cov3D_precomp = cov3D_precomp)
    if len(outputs) == 8:
        rendered_image, radii, rendered_expected_coord, rendered_median_coord, rendered_expected_depth, rendered_median_depth, rendered_alpha, rendered_normal = outputs
    else:
        assert False, "only (depth-)diff-gaussian-rasterization from RaDe-GS supported!"
    # time4 = get_time()
    # print("rasterization:",time4-time3)
    # breakpoint()
    # Those Gaussians that were frustum culled or had a radius of 0 were not visible.
    # They will be excluded from value updates used in the splitting criteria.

    # Debug rendered_image
    # if require_depth:
    # import matplotlib.pyplot as plt
    # image_tensor = rendered_image.permute(1, 2, 0).cpu().detach().numpy()
    # plt.imshow(image_tensor)
    # plt.axis('off')
    # plt.show()

    return {"render": rendered_image,
            "mask": rendered_alpha,
            "expected_coord": rendered_expected_coord,
            "median_coord": rendered_median_coord,
            "expected_depth": rendered_expected_depth,
            "median_depth": rendered_median_depth,
            "viewspace_points": means2D,
            "visibility_filter": radii > 0,
            "radii": radii,
            "normal": rendered_normal,
            "sh_coefs_final": shs_final,
            "extras": extras,
    }
