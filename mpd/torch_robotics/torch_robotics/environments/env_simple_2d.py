import numpy as np
import torch
from matplotlib import pyplot as plt

import torch_robotics.robots as tr_robots
from torch_robotics.environments.env_base import EnvBase
from torch_robotics.environments.primitives import ObjectField, MultiSphereField, MultiBoxField
from torch_robotics.torch_utils.torch_utils import DEFAULT_TENSOR_ARGS
from torch_robotics.visualizers.plot_utils import create_fig_and_axes
import json


def get_pre_obj_list(tensor_args=DEFAULT_TENSOR_ARGS):
    pre_obj_list = [
        MultiSphereField(
            np.array(
                [
                    [-0.43378472328186035, 0.3334643840789795],
                    [0.3313474655151367, 0.6288051009178162],
                    [-0.5656964778900146, -0.484994500875473],
                    [0.42124247550964355, -0.6656165719032288],
                    [0.05636655166745186, -0.5149664282798767],
                    [-0.36961784958839417, -0.12315540760755539],
                    [-0.8740217089653015, -0.4034936726093292],
                    [-0.6359214186668396, 0.6683124899864197],
                    [0.808782160282135, 0.5287870168685913],
                    [-0.023786112666130066, 0.4590069353580475],
                    [0.1455741971731186, 0.16420497000217438],
                    [0.628413736820221, -0.43461447954177856],
                    [0.17965620756149292, -0.8926276564598083],
                    [0.6775968670845032, 0.8817358016967773],
                    [-0.3608766794204712, 0.8313458561897278],
                ]
            ),
            np.array(
                [
                    0.125,
                    0.125,
                    0.125,
                    0.125,
                    0.125,
                    0.125,
                    0.125,
                    0.125,
                    0.125,
                    0.125,
                    0.125,
                    0.125,
                    0.125,
                    0.125,
                    0.125,
                ]
            ),
            tensor_args=tensor_args,
        ),
    ]
    return pre_obj_list


def get_circle_points_list(circle, tensor_args=DEFAULT_TENSOR_ARGS):
    num_points = 15  # 圆周上的点数

    # 利用circle给定的参数生成圆上点
    # 期望circle包含x, y, r字段
    x = circle["x"]
    y = circle["y"]
    r = circle["r"]
    device = tensor_args["device"]
    dtype = tensor_args["dtype"]

    theta = torch.arange(0, num_points, **tensor_args)
    theta = theta * (2 * np.pi / num_points)  # [0, 2π)
    cos_theta = torch.cos(theta)
    sin_theta = torch.sin(theta)
    circle_xy = torch.stack([cos_theta, sin_theta], dim=1)  # [num_points, 2]
    circle_xy = r * circle_xy + torch.tensor([x, y], dtype=dtype, device=device)  # [num_points, 2]

    # 补z=0
    z = torch.zeros((num_points, 1), dtype=dtype, device=device)
    circle_xyz = torch.cat([circle_xy, z], dim=1)  # [num_points, 3]
    circle_xyz = circle_xyz.permute(1, 0).contiguous()  # [3, num_points]
    return circle_xyz


def get_env_simple_2d_obj_list(idx=0, tensor_args=DEFAULT_TENSOR_ARGS):

    with open(
        "/home/ps/桌面/Projects/mpd-splines-public/mpd/torch_robotics/torch_robotics/environments/dispersed_circle_scenes.json",
        "r",
    ) as f:
        scenes = json.load(f)
    obj_list = []
    circle_points_list = torch.tensor([], dtype=torch.float32, device=tensor_args["device"])
    for circle in scenes[idx]:
        obj_list.append(
            MultiSphereField(np.array([[circle["x"], circle["y"]]]), np.array([circle["r"]]), tensor_args=tensor_args)
        )
        circle_points = get_circle_points_list(circle, tensor_args=tensor_args)
        circle_points_list = torch.cat([circle_points_list, circle_points], dim=1)

    # 将circle_points_list填充为[3, 256]维度
    num_points = circle_points_list.shape[1]
    if num_points < 256:
        # 需要在结尾补零
        pad_size = 256 - num_points
        pad_tensor = torch.zeros((3, pad_size), dtype=circle_points_list.dtype, device=circle_points_list.device)
        circle_points_list = torch.cat([circle_points_list, pad_tensor], dim=1)
    elif num_points > 256:
        # 截断多余的点
        circle_points_list = circle_points_list[:, :256]
    return obj_list, circle_points_list


class EnvSimple2D(EnvBase):

    def __init__(self, tensor_args=DEFAULT_TENSOR_ARGS, precompute_sdf_obj_fixed=True, sdf_cell_size=0.005, **kwargs):

        obj_list, obstacle_points_list = get_env_simple_2d_obj_list(idx=5, tensor_args=tensor_args)
        self.obstacle_points_list = obstacle_points_list

        print(f"obstacle_points_list: {obstacle_points_list.shape}")
        # INSERT_YOUR_CODE
        torch.save(
            obstacle_points_list,
            "/home/ps/桌面/Projects/mpd-splines-public/mpd/torch_robotics/torch_robotics/environments/obstacle_points_list.pt",
        )

        super().__init__(
            limits=torch.tensor([[-1, -1], [1, 1]], **tensor_args),  # environments limits
            obj_fixed_list=[ObjectField(obj_list, "dense2d")],
            precompute_sdf_obj_fixed=precompute_sdf_obj_fixed,
            sdf_cell_size=sdf_cell_size,
            tensor_args=tensor_args,
            **kwargs,
        )

    def get_rrt_connect_params(self, robot=None):
        params = dict(n_iters=10000, step_size=0.01, n_radius=0.3, n_pre_samples=50000, max_time=50)

        if isinstance(robot, tr_robots.RobotPointMass2D):
            return params
        else:
            raise NotImplementedError

    def get_gpmp2_params(self, robot=None):
        params = dict(
            n_support_points=64,
            dt=0.04,
            opt_iters=300,
            num_samples=64,
            sigma_start=1e-5,
            sigma_gp=1e-2,
            sigma_goal_prior=1e-5,
            sigma_coll=1e-5,
            step_size=1e-1,
            sigma_start_init=1e-4,
            sigma_goal_init=1e-4,
            sigma_gp_init=0.2,
            sigma_start_sample=1e-4,
            sigma_goal_sample=1e-4,
            solver_params={
                "delta": 1e-2,
                "trust_region": True,
                "method": "cholesky",
            },
        )

        if isinstance(robot, tr_robots.RobotPointMass2D):
            return params
        else:
            raise NotImplementedError

    def get_chomp_params(self, robot=None):
        params = dict(
            n_support_points=64,
            dt=0.04,
            opt_iters=1,  # Keep this 1 for visualization
            weight_prior_cost=1e-4,
            step_size=0.05,
            grad_clip=0.05,
            sigma_start_init=0.001,
            sigma_goal_init=0.001,
            sigma_gp_init=0.3,
            pos_only=False,
        )

        if isinstance(robot, tr_robots.RobotPointMass2D):
            return params
        else:
            raise NotImplementedError


class EnvSimple2DExtraObjectsV00(EnvSimple2D):

    def __init__(self, tensor_args=DEFAULT_TENSOR_ARGS, **kwargs):
        obj_extra_list = [
            MultiSphereField(
                np.array(
                    [
                        [-0.15, 0.15],
                        [-0.075, -0.85],
                        [-0.1, -0.1],
                        # [0.45, -0.1],
                        [0.5, 0.35],
                        [-0.6, -0.85],
                        [0.05, 0.85],
                        [-0.8, 0.15],
                        [0.8, -0.8],
                    ]
                ),
                np.array(
                    [
                        0.05,
                        0.1,
                        0.1,
                        # 0.1,
                        0.1,
                        0.1,
                        0.1,
                        0.1,
                        0.1,
                    ]
                ),
                tensor_args=tensor_args,
            ),
            MultiBoxField(
                np.array(
                    [
                        [0.45, -0.1],
                        [-0.25, -0.5],
                        [0.8, 0.1],
                    ]
                ),
                np.array(
                    [
                        [0.15, 0.25],
                        [0.15, 0.25],
                        [0.15, 0.15],
                    ]
                ),
                tensor_args=tensor_args,
            ),
        ]

        super().__init__(
            obj_extra_list=[ObjectField(obj_extra_list, "dense2d-extraobjects")], tensor_args=tensor_args, **kwargs
        )


if __name__ == "__main__":
    env = EnvSimple2D(precompute_sdf_obj_fixed=True, sdf_cell_size=0.01, tensor_args=DEFAULT_TENSOR_ARGS)
    fig, ax = create_fig_and_axes(env.dim)
    env.render(ax)
    plt.show()

    # Render sdf
    fig, ax = create_fig_and_axes(env.dim)
    env.render_sdf(ax, fig)

    # Render gradient of sdf
    env.render_grad_sdf(ax, fig)
    plt.show()

    #########################################
    env = EnvSimple2DExtraObjectsV00(precompute_sdf_obj_fixed=True, sdf_cell_size=0.01, tensor_args=DEFAULT_TENSOR_ARGS)
    fig, ax = create_fig_and_axes(env.dim)
    env.render(ax)
    plt.show()

    # Render sdf
    fig, ax = create_fig_and_axes(env.dim)
    env.render_sdf(ax, fig)

    # Render gradient of sdf
    env.render_grad_sdf(ax, fig)
    plt.show()
