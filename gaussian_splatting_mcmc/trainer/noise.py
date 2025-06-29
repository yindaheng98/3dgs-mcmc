import torch
from typing import Callable

from gaussian_splatting import GaussianModel
from gaussian_splatting.trainer import AbstractTrainer, TrainerWrapper
from gaussian_splatting.utils import build_scaling_rotation


class Noiser(TrainerWrapper):
    def __init__(
            self, base_trainer: AbstractTrainer,
            noise_lr=5e5,
    ):
        super().__init__(base_trainer)
        self.noise_lr = noise_lr
        assert 'xyz' in self.schedulers, "Noiser requires 'xyz' scheduler to be defined."

    def after_optim_hook(self):
        gaussians = self.model
        xyz_lr = self.schedulers['xyz'](self.curr_step)

        # https://github.com/ubc-vision/3dgs-mcmc/blob/7b4fc9f76a1c7b775f69603cb96e70f80c7e6d13/train.py#L134
        L = build_scaling_rotation(gaussians.get_scaling, gaussians.get_rotation)
        actual_covariance = L @ L.transpose(1, 2)

        def op_sigmoid(x, k=100, x0=0.995):
            return 1 / (1 + torch.exp(-k * (x - x0)))

        noise = torch.randn_like(gaussians._xyz) * (op_sigmoid(1 - gaussians.get_opacity))*self.noise_lr*xyz_lr
        noise = torch.bmm(actual_covariance, noise.unsqueeze(-1)).squeeze(-1)
        gaussians._xyz.add_(noise)

        return super().after_optim_hook()


def NoiseWrapper(
        base_trainer_constructor: Callable[..., AbstractTrainer],
        model: GaussianModel,
        scene_extent: float,
        *args,
        noise_lr=5e5,
        **kwargs) -> Noiser:
    return Noiser(
        base_trainer=base_trainer_constructor(model, scene_extent, *args, **kwargs),
        noise_lr=noise_lr,
    )
