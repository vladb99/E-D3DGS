_base_ = './default.py'

ModelParams = dict(
    disable_filter3D=True,
)

ModelHiddenParams = dict(
    total_num_frames = 5,
)
OptimizationParams = dict(
    maxtime = 5,
    iterations = 55_000,
    densify_until_iter = 55_000,
    position_lr_max_steps = 55_000,
    deformation_lr_max_steps = 55_000,
)