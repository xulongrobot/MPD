import torch
import torch.nn as nn
import numpy as np
from mpd.models.layers.layers import MLP
from mpd.models.diffusion_models.pointnet import PointNet


class ContextModelEEPoseGoal(nn.Module):

    def __init__(self, out_dim=64, n_layers=2, act="relu", **kwargs):
        super().__init__()
        # 9d representation of rotation
        # 3d representation of position
        self.in_dim = 9 + 3
        self.out_dim = out_dim

        # self.net = nn.Identity()
        self.net = MLP(self.in_dim, out_dim, hidden_dim=out_dim, n_layers=n_layers, act=act)

    def forward(self, ee_goal_orientation_normalized, ee_goal_position_normalized, **kwargs):
        pose_repr = torch.cat((ee_goal_orientation_normalized, ee_goal_position_normalized), dim=-1)
        emb = self.net(pose_repr)
        return emb


class ContextModelQs(nn.Module):

    def __init__(self, in_dim, out_dim=64, n_layers=2, act="relu", **kwargs):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim

        # self.net = nn.Identity()
        self.net = MLP(self.in_dim, out_dim, hidden_dim=out_dim, n_layers=n_layers, act=act)

    def forward(self, qs_normalized=None, **kwargs):
        emb = self.net(qs_normalized)
        return emb


class ContextModelENV(nn.Module):
    def __init__(self, in_dim=45, out_dim=64, **kwargs):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.pointnet = PointNet(out_dim)

    def forward(self, environment_obj_list=None, **kwargs):
        """
        environment_obj_list: [batch_size, N_obstacles, 3] -> [batch_size, N_obstacles * 3]
        """

        # print(f"environment_obj_list: {environment_obj_list.shape}")
        x, pos = self.pointnet(environment_obj_list)
        return x


class ContextModelCombined(nn.Module):

    def __init__(
        self, context_model_qs=None, context_model_ee_pose_goal=None, out_dim=64, n_layers=1, act="relu", **kwargs
    ):
        assert not (context_model_qs is None and context_model_ee_pose_goal is None)
        super().__init__()

        self.env_flag = False  # 是否使用环境模型加入感知信息，False则不加入

        self.context_model_qs = context_model_qs
        self.context_model_ee_pose_goal = context_model_ee_pose_goal

        self.in_dim = 0
        if self.context_model_qs is not None:
            self.in_dim += self.context_model_qs.out_dim
        if self.context_model_ee_pose_goal is not None:
            self.in_dim += self.context_model_ee_pose_goal.out_dim

        self.context_model_env = ContextModelENV()
        self.in_dim += self.context_model_env.out_dim

        self.out_dim = out_dim
        self.net = MLP(self.in_dim, self.out_dim, hidden_dim=out_dim, n_layers=n_layers, act=act)

    def forward(
        self,
        qs_normalized=None,
        ee_goal_orientation_normalized=None,
        ee_goal_position_normalized=None,
        environment_obj_list=None,
        **kwargs,
    ):
        emb_q = None
        if self.context_model_qs is not None:
            emb_q = self.context_model_qs(qs_normalized)

        emb_ee_goal_pose = None
        if self.context_model_ee_pose_goal is not None:
            emb_ee_goal_pose = self.context_model_ee_pose_goal(
                ee_goal_orientation_normalized, ee_goal_position_normalized
            )

        if emb_q is not None and emb_ee_goal_pose is not None:
            emb = torch.cat((emb_q, emb_ee_goal_pose), dim=-1)
        elif emb_q is not None:
            emb = emb_q
        elif emb_ee_goal_pose is not None:
            emb = emb_ee_goal_pose

        ### 评估时固定场景的障碍物点云
        # environment_obj_list = torch.load("/home/ps/桌面/Projects/mpd-splines-public/mpd/torch_robotics/torch_robotics/environments/obstacle_points_list.pt")
        # environment_obj_list = environment_obj_list.unsqueeze(0)
        # environment_obj_list = environment_obj_list.repeat(emb.shape[0], 1, 1)

        # print(f"environment_obj_list: {environment_obj_list.shape}")
        emb_env = self.context_model_env(environment_obj_list)
        emb = torch.cat((emb, emb_env), dim=-1)

        context_emb = self.net(emb)
        return context_emb
