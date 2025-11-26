from typing import Callable
from gaussian_splatting import GaussianModel
from gaussian_splatting.dataset import TrainableCameraDataset
from gaussian_splatting.trainer import AbstractDensifier, CameraTrainerWrapper, DepthTrainerWrapper, CameraTrainerWrapper
from .noise import NoiseTrainerWrapper
from .relocate import BaseRelocationTrainer, RelocationTrainerWrapper
from .scale_opacity_reg import ScaleOpacityRegularizeTrainerWrapper

# similar to BaseDensificationTrainer in gaussian_splatting.trainer.densifier.combinations


def BaseNoiseRelocationTrainer(model: GaussianModel, scene_extent: float, *args, **kwargs):
    return NoiseTrainerWrapper(BaseRelocationTrainer, model, scene_extent, *args, **kwargs)


def DepthNoiseRelocationTrainer(model: GaussianModel, scene_extent: float, *args, **kwargs):
    return DepthTrainerWrapper(BaseNoiseRelocationTrainer, model, scene_extent, *args, **kwargs)


NoiseRelocationTrainer = DepthNoiseRelocationTrainer


def CameraNoiseRelocationTrainer(model: GaussianModel, scene_extent: float, dataset: TrainableCameraDataset, *args, **kwargs):
    return CameraTrainerWrapper(
        lambda model, scene_extent, dataset, *args, **kwargs: DepthNoiseRelocationTrainer(model, scene_extent, *args, **kwargs),
        model, scene_extent, dataset,
        *args, **kwargs)


NoRegMCMCTrainer = NoiseRelocationTrainer
NoRegCameraMCMCTrainer = CameraNoiseRelocationTrainer


def BaseScaleOpacityRegularizeNoiseRelocationTrainer(model: GaussianModel, scene_extent: float, *args, **kwargs):
    return ScaleOpacityRegularizeTrainerWrapper(BaseNoiseRelocationTrainer, model, scene_extent, *args, **kwargs)


def DepthScaleOpacityRegularizeNoiseRelocationTrainer(model: GaussianModel, scene_extent: float, *args, **kwargs):
    return DepthTrainerWrapper(BaseScaleOpacityRegularizeNoiseRelocationTrainer, model, scene_extent, *args, **kwargs)


ScaleOpacityRegularizeNoiseRelocationTrainer = DepthScaleOpacityRegularizeNoiseRelocationTrainer


def CameraScaleOpacityRegularizeNoiseRelocationTrainer(model: GaussianModel, scene_extent: float, dataset: TrainableCameraDataset, *args, **kwargs):
    return CameraTrainerWrapper(
        lambda model, scene_extent, dataset, *args, **kwargs: DepthScaleOpacityRegularizeNoiseRelocationTrainer(model, scene_extent, *args, **kwargs),
        model, scene_extent, dataset,
        *args, **kwargs)


MCMCTrainer = ScaleOpacityRegularizeNoiseRelocationTrainer
CameraMCMCTrainer = CameraScaleOpacityRegularizeNoiseRelocationTrainer

# similar to BaseOpacityResetDensificationTrainer in gaussian_splatting.trainer.combinations


def NoiseRelocationTrainerWrapper(
        noargs_base_densifier_constructor: Callable[[GaussianModel, float], AbstractDensifier],
        model: GaussianModel,
        scene_extent: float,
        *args,
        noise_lr=5e5,
        noise_from_iter=0,
        noise_until_iter=29_990,
        **kwargs):
    return NoiseTrainerWrapper(
        lambda model, scene_extent, *args, **kwargs: RelocationTrainerWrapper(
            noargs_base_densifier_constructor, model, scene_extent, *args, **kwargs),
        model, scene_extent,
        *args,
        noise_lr=noise_lr,
        noise_from_iter=noise_from_iter,
        noise_until_iter=noise_until_iter,
        **kwargs)


def ScaleOpacityRegularizeNoiseRelocationTrainerWrapper(
        noargs_base_densifier_constructor: Callable[[GaussianModel, float], AbstractDensifier],
        model: GaussianModel,
        scene_extent: float,
        *args,
        scale_reg_from_iter=0,
        scale_reg_weight=0.01,
        opacity_reg_from_iter=0,
        opacity_reg_weight=0.01,
        **kwargs):
    return ScaleOpacityRegularizeTrainerWrapper(
        NoiseRelocationTrainerWrapper,
        noargs_base_densifier_constructor,
        model, scene_extent,
        *args,
        scale_reg_from_iter=scale_reg_from_iter,
        scale_reg_weight=scale_reg_weight,
        opacity_reg_from_iter=opacity_reg_from_iter,
        opacity_reg_weight=opacity_reg_weight,
        **kwargs)


NoRegMCMCTrainerWrapper = NoiseRelocationTrainerWrapper
MCMCTrainerWrapper = ScaleOpacityRegularizeNoiseRelocationTrainerWrapper

# spacial, similar to DensificationTrainerWrapper in gaussian_splatting.trainer.densifier.combinations
