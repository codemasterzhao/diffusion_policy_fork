from typing import Dict, Tuple
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models as vision_models
from einops import rearrange, reduce
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler

from diffusion_policy.model.common.normalizer import LinearNormalizer
from diffusion_policy.policy.base_lowdim_policy import BaseLowdimPolicy
from diffusion_policy.model.diffusion.transformer_for_diffusion import TransformerForDiffusion
from diffusion_policy.model.diffusion.mask_generator import LowdimMaskGenerator
from diffusion_policy.common.robomimic_config_util import get_robomimic_config
from robomimic.algo import algo_factory
from robomimic.algo.algo import PolicyAlgo
import robomimic.utils.obs_utils as ObsUtils
from robomimic.models.base_nets import SpatialSoftmax
# try:
#     import robomimic.models.base_nets as rmbn
#     if not hasattr(rmbn, 'CropRandomizer'):
#         raise ImportError("CropRandomizer is not in robomimic.models.base_nets")
# except ImportError:
#     # import robomimic.models.obs_core as rmbn
# # import diffusion_policy.model.vision.crop_randomizer as dmvc
from diffusion_policy.common.pytorch_util import dict_apply, replace_submodules

import numpy as np
import itertools
# from e2cnn import gspaces, nn
from einops import rearrange, repeat
# from diffusion_policy.model.equi.equi_3stack_lowdim_encoder import Equivariant3StackLowDimEnc
# from diffusion_policy.model.equi.equi_transformer_action import EquivariantActionTransformer
from diffusion_policy.model.common.rotation_transformer import RotationTransformer
from diffusion_policy.model.lift_mlp import LiftLowDimMLP
from diffusion_policy.model.mlp import SimpleMLP
from scipy.spatial.transform import Rotation
class EquiLowDimPolicy(BaseLowdimPolicy):
    def __init__(self,
            # task params
            horizon,
            n_action_steps,
            n_obs_steps,
            # arch

            # parameters passed to step
            **kwargs):
        super().__init__()


        self.act_net = LiftLowDimMLP()
        print("ActNet params: %e" % sum(p.numel() for p in self.act_net.parameters()))

        self.normalizer = LinearNormalizer()
        self.horizon = horizon
        self.n_action_steps = n_action_steps
        self.n_obs_steps = n_obs_steps

        self.kwargs = kwargs

        self.canonical_to_world = torch.inverse(torch.Tensor([[1, 0, 0], [0, -1, 0], [0, 0, -1]]))
        self.axisangle_to_matrix = RotationTransformer('axis_angle', 'matrix')
        self.quaternion_to_matrix = RotationTransformer('quaternion', 'matrix')
        self.axisangle_to_euler = RotationTransformer('axis_angle', 'euler_angles', to_convention='ZYX')
        self.quaternion_to_euler = RotationTransformer('quaternion', 'euler_angles',to_convention='ZYX')
        # self.eular_to_axisangle = RotationTransformer('euler_angles', 'axis_angle')

    def getRelativeAxisAngle(self, axis_angle_world, ee_quat):
        # quat: x y z w
        # RotationTransformer takes w x y z
        wTg = self.quaternion_to_matrix.forward(ee_quat[:, :, [3, 0, 1, 2]])
        Tc = self.axisangle_to_matrix.forward(axis_angle_world)
        gTgp = torch.transpose(wTg, -1, -2) @ Tc @ wTg
        return self.axisangle_to_matrix.inverse(gTgp)

    def getAbsoluteAxisAngle(self, axis_angle_ee, ee_quat):
        wTg = self.quaternion_to_matrix.forward(ee_quat[:, :, [3, 0, 1, 2]])
        gTgp = self.axisangle_to_matrix.forward(axis_angle_ee)
        Tc = wTg @ gTgp @ torch.transpose(wTg, -1, -2)
        return self.axisangle_to_matrix.inverse(Tc)

    # ========= training  ============
    def set_normalizer(self, normalizer: LinearNormalizer):
        self.normalizer.load_state_dict(normalizer.state_dict())

    def get_optimizer(
            self,
            mlp_weight_decay: float,
            learning_rate: float,
            betas: Tuple[float, float]
        ) -> torch.optim.Optimizer:

        optim_groups = []
        optim_groups.append({
            "params": self.act_net.parameters(),
            "weight_decay": mlp_weight_decay
        })
        optimizer = torch.optim.AdamW(
            optim_groups, lr=learning_rate, betas=betas
        )
        return optimizer
    def preprocess(self,obs):
        obs_obj_xyz=obs[:,0:3]
        obs_obj_relative_xyz=obs[:,7:10]
        obs_obj_qua=obs[:,3:7]
        robo_xyz=obs[:,10:13]
        robo_quo=obs[:,13:17]
        robo_qpos=obs[:,17:19]
        obs_obj_ea=self.quat_to_ea(obs_obj_qua)
        cosines_obj = torch.cos(obs_obj_ea)
        sines_obj = torch.sin(obs_obj_ea)
        vec_obj = torch.empty(obs_obj_ea.shape[0], 6,device=obs_obj_ea.device)
        vec_obj[:, 0::2] = cosines_obj  
        vec_obj[:, 1::2] = sines_obj 
        obs_robo_ea=self.quat_to_ea(robo_quo)
        cosines_robo = torch.cos(obs_robo_ea)
        sines_robo = torch.sin(obs_robo_ea)
        vec_robo = torch.empty(obs_robo_ea.shape[0], 6,device=obs_robo_ea.device)
        vec_robo[:, 0::2] = cosines_robo 
        vec_robo[:, 1::2] = sines_robo 
        obs_features = torch.cat((obs_obj_xyz,vec_obj,obs_obj_relative_xyz,robo_xyz,vec_robo,robo_qpos),dim=1)
        return obs_features
    def post_process(self,action_pred_vector,quat):
        action_pred_vector=torch.squeeze(action_pred_vector)
        quat=torch.squeeze(quat)
        action_pred=torch.empty(action_pred_vector.shape[0],7,device=action_pred_vector.device)
        action_pred[:,0:3]=action_pred_vector[:,0:3]
        ea_vector=action_pred_vector[:,3:9]
        cos_values = ea_vector[:, 0::2]
        sin_values = ea_vector[:, 1::2]
        ea=torch.atan2(sin_values, cos_values)
        quat=torch.squeeze(quat)
        axis_a=self.ab_ea_to_relative_aa(ea,quat)
        action_pred[:,3:6]=axis_a
        action_pred[:,6:7]=action_pred_vector[:,9:10]
        action_pred = repeat(action_pred, 'b d -> b t d', t=self.n_action_steps)
        return action_pred
    def post_process_for_loss(self,action_pred_vector,quat):
        action_pred=torch.empty(action_pred_vector.shape[0],8,device=action_pred_vector.device)
        action_pred[:,0:3]=action_pred_vector[:,0:3]
        ea_vector=action_pred_vector[:,3:9]
        cos_values = ea_vector[:, 0::2]
        sin_values = ea_vector[:, 1::2]
        ea=torch.atan2(sin_values, cos_values)
        quat=self.ea_to_quat(ea)
        action_pred[:,3:7]=quat
        action_pred[:,7:8]=action_pred_vector[:,9:10]
        action_pred = repeat(action_pred, 'b d -> b t d', t=self.n_action_steps)
        return action_pred
    def dataset_to_pred_format(self,dataset,quat):
        dataset=torch.squeeze(dataset[:,0:3])
        quat=torch.squeeze(quat)
        action_xyz=dataset[:,0:3]
        action_axis_angle=dataset[:,3:6]
        action_grasp=dataset[:,6:7]

        # Convert to NumPy array
        action_axis_angle_np = action_axis_angle.to('cpu').numpy()
        quat_np=quat.to('cpu').numpy()
        # Initialize an array to hold the axis-angle representations
        ea_np = np.zeros((action_axis_angle_np.shape[0],3))
        assert action_axis_angle_np.shape[0]==quat_np.shape[0]
        # Process each set of Euler angles
        for i in range(action_axis_angle_np.shape[0]):
            gtgp_rotation = Rotation.from_rotvec( action_axis_angle_np[i])
            wtg_rotation=Rotation.from_quat(quat_np[i])
            gtgp=gtgp_rotation.as_matrix()
            wtg=wtg_rotation.as_matrix()
            wtgp=gtgp@wtg
            tc= np.array([[1, 0, 0],
                            [0, -1, 0],
                            [0, 0, -1]])
            wtc=tc@wtgp
            abs_ea_rotation=Rotation.from_matrix(wtc)
            ea_np[i] = abs_ea_rotation.as_euler('zyx')

        ea_tensor = torch.from_numpy(ea_np).to(dataset.device)
        ea_vector_tensor=torch.empty(ea_tensor.shape[0], 6,device=dataset.device)
        cosines_obj = torch.cos(ea_tensor)
        sines_obj = torch.sin(ea_tensor)
        ea_vector_tensor[:, 0::2] = cosines_obj  
        ea_vector_tensor[:, 1::2] = sines_obj 
        action_features = torch.cat((action_xyz,ea_vector_tensor,action_grasp),dim=1)
        action_features = repeat(action_features, 'b d -> b t d', t=self.n_action_steps)
        return action_features
    def ab_ea_to_relative_aa(self,ab_ea,quat):
        device=ab_ea.device
        assert ab_ea.shape[0]==quat.shape[0]
        ab_ea_np = ab_ea.to('cpu').numpy()
        quat_np=quat.to('cpu').numpy()
        aa_np=np.zeros((ab_ea.shape[0],3))
        for i in range(aa_np.shape[0]):
            wtc_rotation = Rotation.from_euler( 'zyx',ab_ea_np[i])
            wtg_rotation=Rotation.from_quat(quat_np[i])
            tc= np.array([[1, 0, 0],
                            [0, -1, 0],
                            [0, 0, -1]])
            wtc=wtc_rotation.as_matrix()
            wtgp=tc@wtc
            wtg=wtg_rotation.as_matrix()
            gtw=np.linalg.inv(wtg)
            gtgp=wtgp@gtw
            relative_rotation=Rotation.from_matrix(gtgp)
            aa_np[i] = relative_rotation.as_rotvec()
        aa=torch.from_numpy(aa_np).to(device=device)
        aa=torch.squeeze(aa)
        return aa
    def quat_to_ea(self,quat):
        device=quat.device
        quaternion_np = quat.to('cpu').numpy()
        euler_angles_np=np.zeros((quaternion_np.shape[0],3))
        for i in range(quaternion_np.shape[0]):
            rotation = Rotation.from_quat( quaternion_np[i])
            euler_angle = rotation.as_euler('zyx')
            euler_angles_np[i] = euler_angle
        # Use SciPy to convert from quaternion to Euler angles

        # Convert back to PyTorch tensor
        euler_angles_tensor = torch.from_numpy(euler_angles_np).to(device=device)
        return euler_angles_tensor
    def ea_to_quat(self,ea):
        device=ea.device
        ea_np = ea.to('cpu').numpy()
        quat_np=np.zeros((ea_np.shape[0],4))
        for i in range(ea_np.shape[0]):
            rotation = Rotation.from_euler( 'zyx',ea_np[i])
            quat = rotation.as_quat()
            quat_np[i] = quat
        # Use SciPy to convert from quaternion to Euler angles

        # Convert back to PyTorch tensor
        quat_tensor = torch.from_numpy(quat_np).to(device=device)
        return quat_tensor
    def ea_to_axis_angle(self,ea):

    # Convert to NumPy array
        device=ea.device
        euler_angles_np = ea.to('cpu').numpy()

        # Initialize an array to hold the axis-angle representations
        axis_angles_np = np.zeros_like(euler_angles_np)

        # Process each set of Euler angles
        for i in range(euler_angles_np.shape[0]):
            rotation = Rotation.from_euler('zyx', euler_angles_np[i])
            axis_angle = rotation.as_rotvec()
            axis_angles_np[i] = axis_angle

        # Convert back to PyTorch tensor
        axis_angle_tensor = torch.from_numpy(axis_angles_np).to(device)
        return axis_angle_tensor
    # ========= inference  ============
    def predict_action(self, obs_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        obs_dict: must include "obs" key
        result: must include "action" key
        """
        assert 'past_action' not in obs_dict # not implemented yet
        assert self.n_obs_steps==1
        obs = obs_dict['obs']
        obs = obs[:,0,:]
        obs=torch.squeeze(obs)
        robo_quo=obs[:,13:17]
        obs_features=self.preprocess(obs)
        # action_pred_vector = self.act_net(obs_features).tensor
        action_pred_vector = self.act_net(obs_features)
        action_pred=self.post_process(action_pred_vector,robo_quo)

        action = action_pred
        result = {
            'action': action,
            'action_pred': action_pred
        }
        return result

    # ========= training  ============
    def set_normalizer(self, normalizer: LinearNormalizer):
        self.normalizer.load_state_dict(normalizer.state_dict())

    def compute_loss(self, batch):
        assert 'valid_mask' not in batch
        assert self.n_obs_steps==1
        assert self.n_action_steps==1
        obs = batch['obs']
        obs = obs[:,0,:]
        obs=torch.squeeze(obs)
        actions = batch['action'].clone()
        robo_quo=obs[:,13:17]
        ee_quat = obs[:, 3:7]
        ee_quat = repeat(ee_quat, 'b d -> b t d', t=self.n_action_steps)

        # actions[:, :, 3:6] = self.getRelativeAxisAngle(actions[:, :, 3:6], ee_quat)

        obs_features = self.preprocess(obs)
        # action_pred_vector = self.act_net(obs_features).tensor
        action_pred_vector = self.act_net(obs_features)
        action_pred_vector = repeat(action_pred_vector, 'b d -> b t d', t=self.n_action_steps)
        start = self.n_obs_steps - 1
        end = start + self.n_action_steps
        loss = F.mse_loss(action_pred_vector, self.dataset_to_pred_format(actions[:, start:end],robo_quo))
        return loss
    def reset(self):
        return
        