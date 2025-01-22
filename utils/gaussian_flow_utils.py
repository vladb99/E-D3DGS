import numpy as np
import torch

def predict(render_pkg_t1, render_pkg_t2):
    # Gaussian parameters at t_1
    proj_2D_t_1 = render_pkg_t1["proj_2D"]
    gs_per_pixel = render_pkg_t1["gs_per_pixel"].long()
    weight_per_gs_pixel = render_pkg_t1["weight_per_gs_pixel"]
    x_mu = render_pkg_t1["x_mu"]
    cov2D_inv_t_1 = render_pkg_t1["conic_2D"].detach()

    # Gaussian parameters at t_2
    proj_2D_t_2 = render_pkg_t2["proj_2D"]
    cov2D_inv_t_2 = render_pkg_t2["conic_2D"]
    cov2D_t_2 = render_pkg_t2["conic_2D_inv"]

    cov2D_t_2_mtx = torch.zeros([cov2D_t_2.shape[0], 2, 2]).cuda()
    cov2D_t_2_mtx[:, 0, 0] = cov2D_t_2[:, 0]
    cov2D_t_2_mtx[:, 0, 1] = cov2D_t_2[:, 1]
    cov2D_t_2_mtx[:, 1, 0] = cov2D_t_2[:, 1]
    cov2D_t_2_mtx[:, 1, 1] = cov2D_t_2[:, 2]

    cov2D_inv_t_1_mtx = torch.zeros([cov2D_inv_t_1.shape[0], 2, 2]).cuda()
    cov2D_inv_t_1_mtx[:, 0, 0] = cov2D_inv_t_1[:, 0]
    cov2D_inv_t_1_mtx[:, 0, 1] = cov2D_inv_t_1[:, 1]
    cov2D_inv_t_1_mtx[:, 1, 0] = cov2D_inv_t_1[:, 1]
    cov2D_inv_t_1_mtx[:, 1, 1] = cov2D_inv_t_1[:, 2]

    # B_t_2
    U_t_2 = torch.svd(cov2D_t_2_mtx)[0]
    S_t_2 = torch.svd(cov2D_t_2_mtx)[1]
    V_t_2 = torch.svd(cov2D_t_2_mtx)[2]
    B_t_2 = torch.bmm(torch.bmm(U_t_2, torch.diag_embed(S_t_2)**(1/2)), V_t_2.transpose(1,2))

    # B_t_1 ^(-1)
    U_inv_t_1 = torch.svd(cov2D_inv_t_1_mtx)[0]
    S_inv_t_1 = torch.svd(cov2D_inv_t_1_mtx)[1]
    V_inv_t_1 = torch.svd(cov2D_inv_t_1_mtx)[2]
    B_inv_t_1 = torch.bmm(torch.bmm(U_inv_t_1, torch.diag_embed(S_inv_t_1)**(1/2)), V_inv_t_1.transpose(1,2))

    # calculate B_t_2*B_inv_t_1
    B_t_2_B_inv_t_1 = torch.bmm(B_t_2, B_inv_t_1)

    # full formulation of GaussianFlow
    cov_multi = (B_t_2_B_inv_t_1[gs_per_pixel] @ x_mu.permute(0,2,3,1).unsqueeze(-1).detach()).squeeze()
    predicted_flow_by_gs = (cov_multi + proj_2D_t_2[gs_per_pixel] - proj_2D_t_1[gs_per_pixel].detach() - x_mu.permute(0,2,3,1).detach()) * weight_per_gs_pixel.detach().unsqueeze(-1)

    return predicted_flow_by_gs