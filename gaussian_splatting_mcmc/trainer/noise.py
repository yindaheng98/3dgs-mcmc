import torch
from typing import Callable

from gaussian_splatting import GaussianModel
from gaussian_splatting.dataset import CameraDataset
from gaussian_splatting.trainer import AbstractTrainer, TrainerWrapper
from gaussian_splatting.utils import build_scaling_rotation
from gaussian_splatting.trainer import BaseTrainer


class Noiser(TrainerWrapper):
    def __init__(
            self, base_trainer: AbstractTrainer,
            noise_lr=5e5,
            noise_from_iter=0,
            noise_until_iter=29_990,
    ):
        super().__init__(base_trainer)
        self.noise_lr = noise_lr
        assert 'xyz' in self.schedulers, "Noiser requires 'xyz' scheduler to be defined."
        self.noise_from_iter = noise_from_iter
        self.noise_until_iter = noise_until_iter

    def after_optim_hook(self, *args, **kwargs):
        if self.noise_from_iter <= self.curr_step <= self.noise_until_iter:
            with torch.no_grad():
                gaussians = self.model
                xyz_lr = self.schedulers['xyz'](self.curr_step)
                self.add_noise(gaussians, xyz_lr)
        return super().after_optim_hook(*args, **kwargs)

    def add_noise(self, gaussians: GaussianModel, xyz_lr: float):

        # https://github.com/ubc-vision/3dgs-mcmc/blob/7b4fc9f76a1c7b775f69603cb96e70f80c7e6d13/train.py#L134
        L = build_scaling_rotation(gaussians.get_scaling, gaussians.get_rotation)
        actual_covariance = L @ L.transpose(1, 2)

        def op_sigmoid(x, k=100, x0=0.995):
            return 1 / (1 + torch.exp(-k * (x - x0)))

        noise = torch.randn_like(gaussians._xyz) * (op_sigmoid(1 - gaussians.get_opacity))*self.noise_lr*xyz_lr
        noise = torch.bmm(actual_covariance, noise.unsqueeze(-1)).squeeze(-1)
        gaussians._xyz.add_(noise)


def NoiseTrainerWrapper(
        base_trainer_constructor: Callable[..., AbstractTrainer],
        model: GaussianModel,
        dataset: CameraDataset,
        *args,
        noise_lr=5e5,
        noise_from_iter=0,
        noise_until_iter=29_990,
        **configs) -> Noiser:
    return Noiser(
        base_trainer=base_trainer_constructor(model, dataset, *args, **configs),
        noise_lr=noise_lr,
        noise_from_iter=noise_from_iter,
        noise_until_iter=noise_until_iter,
    )


def BaseNoiseTrainer(model: GaussianModel, dataset: CameraDataset, **configs) -> Noiser:
    return NoiseTrainerWrapper(BaseTrainer, model, dataset, **configs)

# similar to gaussian_splatting.trainer.depth
