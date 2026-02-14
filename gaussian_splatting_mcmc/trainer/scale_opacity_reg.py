
from typing import Callable
import torch

from gaussian_splatting import Camera, GaussianModel
from gaussian_splatting.dataset import CameraDataset
from gaussian_splatting.trainer import AbstractTrainer, TrainerWrapper, BaseTrainer


class ScaleOpacityRegularizer(TrainerWrapper):
    def __init__(
            self, base_trainer: AbstractTrainer,
            scale_reg_from_iter=0,
            scale_reg_weight=0.01,
            opacity_reg_from_iter=0,
            opacity_reg_weight=0.01,
    ):
        super().__init__(base_trainer)
        self.scale_reg_from_iter = scale_reg_from_iter
        self.scale_reg_weight = scale_reg_weight
        self.opacity_reg_from_iter = opacity_reg_from_iter
        self.opacity_reg_weight = opacity_reg_weight

    def loss(self, out: dict, camera: Camera) -> torch.Tensor:
        if self.curr_step < self.opacity_reg_from_iter:
            return super().loss(out, camera)
        loss = super().loss(out, camera)
        loss += self.scale_reg_weight * torch.abs(self.model.get_scaling).mean()
        loss += self.opacity_reg_weight * torch.abs(self.model.get_opacity).mean()
        return loss


def ScaleOpacityRegularizeTrainerWrapper(
    base_trainer_constructor: Callable[..., AbstractTrainer],
    model: GaussianModel,
    dataset: CameraDataset,
    *args,
    scale_reg_from_iter=0,
    scale_reg_weight=0.01,
    opacity_reg_from_iter=0,
    opacity_reg_weight=0.01,
    **configs
) -> ScaleOpacityRegularizer:
    return ScaleOpacityRegularizer(
        base_trainer_constructor(model, dataset, *args, **configs),
        scale_reg_from_iter=scale_reg_from_iter,
        scale_reg_weight=scale_reg_weight,
        opacity_reg_from_iter=opacity_reg_from_iter,
        opacity_reg_weight=opacity_reg_weight,
    )


def BaseScaleOpacityRegularizeTrainer(model: GaussianModel, dataset: CameraDataset, **configs) -> ScaleOpacityRegularizer:
    return ScaleOpacityRegularizeTrainerWrapper(BaseTrainer, model, dataset, **configs)

# similar to gaussian_splatting.trainer.depth
