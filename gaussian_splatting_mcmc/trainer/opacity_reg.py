
import torch

from gaussian_splatting import Camera, GaussianModel
from gaussian_splatting.trainer import AbstractTrainer, TrainerWrapper, BaseTrainer


class OpacityRegularizer(TrainerWrapper):
    def __init__(
            self, base_trainer: AbstractTrainer,
            opacity_reg_from_iter=0,
            opacity_reg_weight=0.01,
    ):
        super().__init__(base_trainer)
        self.opacity_reg_from_iter = opacity_reg_from_iter
        self.opacity_reg_weight = opacity_reg_weight

    def loss(self, out: dict, camera: Camera) -> torch.Tensor:
        if self.curr_step < self.opacity_reg_from_iter:
            return super().loss(out, camera)
        return super().loss(out, camera) + self.opacity_reg_weight * torch.abs(self.model.get_opacity).mean()


def OpacityRegularizeTrainerWrapper(
    base_constructor,
    model: GaussianModel,
    scene_extent: float,
    *args,
    opacity_reg_from_iter=0,
    opacity_reg_weight=0.01,
    **kwargs
) -> OpacityRegularizer:
    return OpacityRegularizer(
        base_constructor(
            model,
            scene_extent,
            *args, **kwargs
        ),
        opacity_reg_from_iter=opacity_reg_from_iter,
        opacity_reg_weight=opacity_reg_weight,
    )


def BaseOpacityRegularizeTrainer(model: GaussianModel, scene_extent: float, *args, **kwargs) -> OpacityRegularizer:
    return OpacityRegularizeTrainerWrapper(BaseTrainer, model, scene_extent, *args, **kwargs)
