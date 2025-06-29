
import torch

from gaussian_splatting import Camera, GaussianModel
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
    base_constructor,
    model: GaussianModel,
    *args,
    scale_reg_from_iter=0,
    scale_reg_weight=0.01,
    opacity_reg_from_iter=0,
    opacity_reg_weight=0.01,
    **kwargs
) -> ScaleOpacityRegularizer:
    return ScaleOpacityRegularizer(
        base_constructor(
            model,
            *args, **kwargs
        ),
        scale_reg_from_iter=scale_reg_from_iter,
        scale_reg_weight=scale_reg_weight,
        opacity_reg_from_iter=opacity_reg_from_iter,
        opacity_reg_weight=opacity_reg_weight,
    )


def BaseScaleOpacityRegularizeTrainer(model: GaussianModel, scene_extent: float, *args, **kwargs) -> ScaleOpacityRegularizer:
    return ScaleOpacityRegularizeTrainerWrapper(BaseTrainer, model, scene_extent, *args, **kwargs)
