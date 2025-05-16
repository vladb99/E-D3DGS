import pyvista as pv
import numpy as np

mesh_file = '/home/vbratulescu/git/experiments/dynamic/Nersemble/407/TongueMaskTongueEmbeddingTongueSampling/tetrahedra_meshes/ours_80000/timestep_90/recon.ply'
#mesh_file = '/home/vbratulescu/git/experiments/dynamic/Nersemble/037/TongueBest/tetrahedra_meshes/ours_80000/timestep_90/recon.ply'
mesh = pv.read(mesh_file)
plotter = pv.Plotter(window_size=(550, 802))
plotter.add_mesh(mesh, color='white')

views = {
    "central":  np.array([(1.6683063481353217, -0.7333251078291683, 1.931589953158265),
 (0.09087288809784043, 0.09789377846425318, -0.07711282857679756),
 (-0.26515856676978106, -0.9465870651442991, -0.18347714454421335)]
),
    "side":     np.array([(2.298656881844349, -0.15459584433698387, -1.5857701291130304),
 (0.09087288809784043, 0.09789377846425318, -0.07711282857679756),
 (-0.24215929893047983, -0.9503723441857076, -0.1953235299406126)]
),
    "thongue":  np.array([(0.9933165509450862, -0.35115023159878744, 0.14970679883384724),
 (0.1036945355663955, 0.13272481887906912, -0.06655414261018643),
 (-0.4307695341871955, -0.8804752858259745, -0.19799211970635394)]
),
}

plotter.camera_position = views['thongue']
plotter.camera.zoom(3)
plotter.show(auto_close=False)
print(plotter.camera_position)

