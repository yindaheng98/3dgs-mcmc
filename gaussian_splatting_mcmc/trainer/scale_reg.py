
import torch

from gaussian_splatting import Camera, GaussianModel
from gaussian_splatting.trainer import AbstractTrainer, TrainerWrapper, BaseTrainer


class ScaleRegularizer(TrainerWrapper):
    def __init__(
            self, base_trainer: AbstractTrainer,
            scale_reg_from_iter=0,
            scale_reg_weight=0.01,
    ):
        super().__init__(base_trainer)
        self.scale_reg_from_iter = scale_reg_from_iter
        self.scale_reg_weight = scale_reg_weight

    def loss(self, out: dict, camera: Camera) -> torch.Tensor:
        if self.curr_step < self.scale_reg_from_iter:
            return super().loss(out, camera)
        return super().loss(out, camera) + self.scale_reg_weight * torch.abs(self.model.get_scaling).mean()


def ScaleRegularizeTrainerWrapper(
    base_constructor,
    model: GaussianModel,
    scene_extent: float,
    *args,
    scale_reg_from_iter=0,
    scale_reg_weight=0.01,
    **kwargs
) -> ScaleRegularizer:
    return ScaleRegularizer(
        base_constructor(
            model,
            scene_extent,
            *args, **kwargs
        ),
        scale_reg_from_iter=scale_reg_from_iter,
        scale_reg_weight=scale_reg_weight,
    )


def BaseScaleRegularizeTrainer(model: GaussianModel, scene_extent: float, *args, **kwargs) -> ScaleRegularizer:
    return ScaleRegularizeTrainerWrapper(BaseTrainer, model, scene_extent, *args, **kwargs)
