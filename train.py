"""
Usage:
Training:
python train.py --config-name=train_diffusion_lowdim_workspace

python train.py --config-dir=. --config-name=train_robomimic_lowdim_workspace.yaml training.seed=42 training.device=cuda:0 hydra.run.dir='data/outputs/${now:%Y.%m.%d}/${now:%H.%M.%S}_${name}_${task_name}'
"""


import sys
# use line-buffering for both stdout and stderr
sys.stdout = open(sys.stdout.fileno(), mode='w', buffering=1)
sys.stderr = open(sys.stderr.fileno(), mode='w', buffering=1)

import hydra
from omegaconf import OmegaConf
import pathlib
import robosuite 

# allows arbitrary python code execution in configs using the ${eval:''} resolver
OmegaConf.register_new_resolver("eval", eval, replace=True)

@hydra.main(
    version_base=None,
    config_path=str(pathlib.Path(__file__).parent.joinpath(
        'diffusion_policy','config'))
)
def main(cfg: OmegaConf):
    # resolve immediately so all the ${now:} resolvers
    # will use the same time.
    OmegaConf.resolve(cfg)

    print(robosuite.__version__)
    cls = hydra.utils.get_class(cfg._target_)
    workspace: TrainRobomimicLowdimWorkspace = cls(cfg)
    workspace.run()

if __name__ == "__main__":
    main()
