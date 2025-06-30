from typing import Callable
from gaussian_splatting import GaussianModel
from gaussian_splatting.dataset import CameraDataset, TrainableCameraDataset
from gaussian_splatting.trainer import AbstractDensifier, CameraTrainerWrapper, DepthTrainerWrapper, CameraTrainerWrapper
from .noise import NoiseWrapper
from .relocate import Relocater, BaseRelocationTrainer
from .scale_opacity_reg import ScaleOpacityRegularizeTrainerWrapper


def NoiseRelocationDensifierTrainerWrapper(
        noargs_base_densifier_constructor: Callable[[GaussianModel, float], AbstractDensifier],
        model: GaussianModel,
        scene_extent: float,
        *args,
        noise_lr=5e5,
        **kwargs):
    return NoiseWrapper(
        lambda model, scene_extent, *args, **kwargs: Relocater(model, scene_extent, noargs_base_densifier_constructor(model, scene_extent), *args, **kwargs),
        model, scene_extent,
        *args,
        noise_lr=noise_lr,
        **kwargs)


def ScaleOpacityRegularizeNoiseRelocationDensifierTrainerWrapper(
        noargs_base_densifier_constructor: Callable[[GaussianModel, float], AbstractDensifier],
        model: GaussianModel,
        *args,
        scale_reg_from_iter=0,
        scale_reg_weight=0.01,
        opacity_reg_from_iter=0,
        opacity_reg_weight=0.01,
        **kwargs):
    return ScaleOpacityRegularizeTrainerWrapper(
        NoiseRelocationDensifierTrainerWrapper,
        noargs_base_densifier_constructor, model,
        *args,
        scale_reg_from_iter=scale_reg_from_iter,
        scale_reg_weight=scale_reg_weight,
        opacity_reg_from_iter=opacity_reg_from_iter,
        opacity_reg_weight=opacity_reg_weight,
        **kwargs)


MCMCDensifierTrainerWrapper = ScaleOpacityRegularizeNoiseRelocationDensifierTrainerWrapper


def NoiseRelocationTrainer(model: GaussianModel, scene_extent: float, *args, **kwargs):
    return NoiseWrapper(BaseRelocationTrainer, model, scene_extent, *args, **kwargs)


def DepthNoiseRelocationTrainer(model: GaussianModel, scene_extent: float, *args, **kwargs):
    return DepthTrainerWrapper(NoiseRelocationTrainer, model, scene_extent, *args, **kwargs)


NoRegMCMCTrainer = DepthNoiseRelocationTrainer


def BaseMCMCTrainer(model: GaussianModel, scene_extent: float, *args, **kwargs):
    return ScaleOpacityRegularizeTrainerWrapper(NoiseRelocationTrainer, model, scene_extent, *args, **kwargs)


def DepthMCMCTrainer(model: GaussianModel, scene_extent: float, *args, **kwargs):
    return DepthTrainerWrapper(BaseMCMCTrainer, model, scene_extent, *args, **kwargs)


MCMCTrainer = DepthMCMCTrainer


def CameraMCMCTrainer(model: GaussianModel, scene_extent: float, dataset: TrainableCameraDataset, *args, **kwargs):
    return CameraTrainerWrapper(
        lambda model, scene_extent, dataset, *args, **kwargs: DepthMCMCTrainer(model, scene_extent, *args, **kwargs),
        model, scene_extent, dataset,
        *args, **kwargs)


def CameraNoRegMCMCTrainer(model: GaussianModel, scene_extent: float, dataset: TrainableCameraDataset, *args, **kwargs):
    return CameraTrainerWrapper(
        lambda model, scene_extent, dataset, *args, **kwargs: DepthNoiseRelocationTrainer(model, scene_extent, *args, **kwargs),
        model, scene_extent, dataset,
        *args, **kwargs)
