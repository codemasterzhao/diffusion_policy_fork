from typing import Dict, Tuple
import torch
import torch.nn as nn
from omegaconf import OmegaConf
import torch.nn.functional as F

from diffusion_policy.model.common.normalizer import LinearNormalizer
from diffusion_policy.policy.base_lowdim_policy import BaseLowdimPolicy
from diffusion_policy.model.bet.action_ae.discretizers.k_means import KMeansDiscretizer
from diffusion_policy.model.bet.latent_generators.mingpt import MinGPT
from diffusion_policy.model.bet.utils import eval_mode
from diffusion_policy.common.robomimic_config_util import get_robomimic_config
from robomimic.algo import algo_factory
from robomimic.algo.algo import PolicyAlgo
import robomimic.utils.obs_utils as ObsUtils
import diffusion_policy.model.vision.crop_randomizer as dmvc
try:
    import robomimic.models.base_nets as rmbn
    if not hasattr(rmbn, 'CropRandomizer'):
        raise ImportError("CropRandomizer is not in robomimic.models.base_nets")
except ImportError:
    import robomimic.models.obs_core as rmbn
from diffusion_policy.common.pytorch_util import dict_apply, replace_submodules
class BETLowdimPolicy(BaseLowdimPolicy):
    def __init__(self, 
            shape_meta:dict,
            action_ae: KMeansDiscretizer, 
            obs_encoding_net: nn.Module, 
            state_prior: MinGPT,
            horizon,
            n_action_steps,
            n_obs_steps,
            crop_shape=None,
            obs_encoder_group_norm=False,
            eval_fixed_crop=False
            ):
        super().__init__()
        action_shape = shape_meta['action']['shape']
        assert len(action_shape) == 1
        action_dim = action_shape[0]
        obs_shape_meta = shape_meta['obs']
        obs_config = {
            'low_dim': [],
            'rgb': [],
            'depth': [],
            'scan': []
        }
        obs_key_shapes = dict()
        for key, attr in obs_shape_meta.items():
            shape = attr['shape']
            obs_key_shapes[key] = list(shape)

            type = attr.get('type', 'low_dim')
            if type == 'rgb':
                obs_config['rgb'].append(key)
            elif type == 'low_dim':
                obs_config['low_dim'].append(key)
            else:
                raise RuntimeError(f"Unsupported obs type: {type}")

        # get raw robomimic config
        config = get_robomimic_config(
            algo_name='bc_rnn',
            hdf5_type='image',
            task_name='square',
            dataset_type='ph')
        
        with config.unlocked():
            # set config with shape_meta
            config.observation.modalities.obs = obs_config

            if crop_shape is None:
                for key, modality in config.observation.encoder.items():
                    if modality.obs_randomizer_class == 'CropRandomizer':
                        modality['obs_randomizer_class'] = None
            else:
                # set random crop parameter
                ch, cw = crop_shape
                for key, modality in config.observation.encoder.items():
                    if modality.obs_randomizer_class == 'CropRandomizer':
                        modality.obs_randomizer_kwargs.crop_height = ch
                        modality.obs_randomizer_kwargs.crop_width = cw

        # init global state
        ObsUtils.initialize_obs_utils_with_config(config)

        # load model
        policy: PolicyAlgo = algo_factory(
                algo_name=config.algo_name,
                config=config,
                obs_key_shapes=obs_key_shapes,
                ac_dim=action_dim,
                device='cpu',
            )

        obs_encoder = policy.nets['policy'].nets['encoder'].nets['obs']
        
        if obs_encoder_group_norm:
            # replace batch norm with group norm
            replace_submodules(
                root_module=obs_encoder,
                predicate=lambda x: isinstance(x, nn.BatchNorm2d),
                func=lambda x: nn.GroupNorm(
                    num_groups=x.num_features//16, 
                    num_channels=x.num_features)
            )
            # obs_encoder.obs_nets['agentview_image'].nets[0].nets
        
        # obs_encoder.obs_randomizers['agentview_image']
        if eval_fixed_crop:
            replace_submodules(
                root_module=obs_encoder,
                predicate=lambda x: isinstance(x, rmbn.CropRandomizer),
                func=lambda x: dmvc.CropRandomizer(
                    input_shape=x.input_shape,
                    crop_height=x.crop_height,
                    crop_width=x.crop_width,
                    num_crops=x.num_crops,
                    pos_enc=x.pos_enc
                )
            )



        self.normalizer = LinearNormalizer()
        self.action_ae = action_ae
        self.obs_encoding_net = obs_encoder
        self.state_prior = state_prior
        self.horizon = horizon
        self.n_action_steps = n_action_steps
        self.n_obs_steps = n_obs_steps

    # ========= inference  ============
    def predict_action(self, obs_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        obs_dict: must include "obs" key
        result: must include "action" key
        """
        if 'obs' in obs_dict:
            obs_dict=obs_dict['obs']
        assert 'agentview_image' in obs_dict

        assert 'past_action' not in obs_dict # not implemented yet
        nobs = self.normalizer.normalize(obs_dict)
        seq=obs_dict['agentview_image'].shape[1]
        batch=obs_dict['agentview_image'].shape[0]
        # if seq!=self.n_obs_steps:
        #     def pad(input_tensor: torch.Tensor) -> torch.Tensor:
        #         new_tensor=input_tensor[:,:self.n_obs_steps,...].reshape(-1, *input_tensor.shape[2:])
        #         return new_tensor
        # if seq==1:
        #         def pad(input_tensor: torch.Tensor) -> torch.Tensor:
        #             shape = list(input_tensor.shape)
        #             shape[1] = self.n_obs_steps-seq
        #             new_tensor=torch.cat([input_tensor,torch.full(shape, -2,device=input_tensor.device)],dim=1).reshape(-1, *input_tensor.shape[2:])
        #             return new_tensor
        # pad To to T
        def pad(input_tensor: torch.Tensor) -> torch.Tensor:
            shape = list(input_tensor.shape)
            shape[1] = self.horizon-self.n_obs_steps
            new_tensor=torch.cat([input_tensor[:,:self.n_obs_steps],torch.full(shape, -2,device=input_tensor.device)],dim=1).reshape(-1, *input_tensor.shape[2:])
            return new_tensor
        # else:
        #     def pad(input_tensor: torch.Tensor) -> torch.Tensor:
        #         new_tensor=input_tensor.reshape(-1, *input_tensor.shape[2:])
        #         return new_tensor

        obs = dict_apply(nobs, pad)

        # (B,T,Do)
        enc_obs = self.obs_encoding_net(obs)
        enc_obs=enc_obs.reshape(batch,self.horizon,-1)
        # Sample latents from the prior
        latents, offsets = self.state_prior.generate_latents(enc_obs)

        # un-descritize
        naction_pred = self.action_ae.decode_actions(
            latent_action_batch=(latents, offsets)
        )
        # (B,T,Da)

        # un-normalize
        action_pred = self.normalizer['action'].unnormalize(naction_pred)

        # get action
        start = self.n_obs_steps - 1
        end = start + self.n_action_steps
        action = action_pred[:,start:end]
        result = {
            'action': action,
            'action_pred': action_pred
        }
        return result

    # ========= training  ============
    def set_normalizer(self, normalizer: LinearNormalizer):
        self.normalizer.load_state_dict(normalizer.state_dict())
    
    def fit_action_ae(self, input_actions: torch.Tensor):
        self.action_ae.fit_discretizer(input_actions=input_actions)
    
    def get_latents(self, latent_collection_loader):
        training_latents = list()
        with eval_mode(self.action_ae, self.obs_encoding_net, no_grad=True):
            for observations, action, mask in latent_collection_loader:
                obs, act = observations.to(self.device, non_blocking=True), action.to(self.device, non_blocking=True)
                enc_obs = self.obs_encoding_net(obs)
                latent = self.action_ae.encode_into_latent(act, enc_obs)
                reconstructed_action = self.action_ae.decode_actions(
                    latent,
                    enc_obs,
                )
                total_mse_loss += F.mse_loss(act, reconstructed_action, reduction="sum")
                if type(latent) == tuple:
                    # serialize into tensor; assumes last dim is latent dim
                    detached_latents = tuple(x.detach() for x in latent)
                    training_latents.append(torch.cat(detached_latents, dim=-1))
                else:
                    training_latents.append(latent.detach())
        training_latents_tensor = torch.cat(training_latents, dim=0)
        return training_latents_tensor

    def get_optimizer(
            self, weight_decay: float, learning_rate: float, betas: Tuple[float, float]
        ) -> torch.optim.Optimizer:
        return self.state_prior.get_optimizer(
                weight_decay=weight_decay, 
                learning_rate=learning_rate, 
                betas=tuple(betas))
    
    def compute_loss(self, batch):
        # normalize input
        assert 'valid_mask' not in batch
        # nbatch = self.normalizer.normalize(batch)
        nobs = self.normalizer.normalize(batch['obs'])
        action = self.normalizer['action'].normalize(batch['action'])
        # action=action[:,:self.n_obs_steps,...]
        # mask out observations after n_obs_steps
        def pad(input_tensor: torch.Tensor) -> torch.Tensor:
            shape = list(input_tensor.shape)
            shape[1] = self.horizon-self.n_obs_steps
            new_tensor=torch.cat([input_tensor[:, :self.n_obs_steps, ...],torch.full(shape, -2,device=input_tensor.device)],dim=1).reshape(-1, *input_tensor.shape[2:])
            return new_tensor
        # def pad(input_tensor: torch.Tensor) -> torch.Tensor:
        #     new_tensor=input_tensor[:,:self.n_obs_steps,...].reshape(-1, *input_tensor.shape[2:])
        #     return new_tensor
        this_nobs = dict_apply(nobs, 
            pad
)
        enc_obs = self.obs_encoding_net(this_nobs)
        latent = self.action_ae.encode_into_latent(action, enc_obs)
        enc_obs=enc_obs.reshape(action.shape[0],self.horizon,-1)
        _, loss, loss_components = self.state_prior.get_latent_and_loss(
            obs_rep=enc_obs,
            target_latents=latent,
            return_loss_components=True,
        )
        return loss, loss_components
