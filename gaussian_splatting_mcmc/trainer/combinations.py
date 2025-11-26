from functools import partial
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
        lambda model, scene_extent, dataset, *args, **kwargs: NoiseRelocationTrainer(model, scene_extent, *args, **kwargs),
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
        lambda model, scene_extent, dataset, *args, **kwargs: ScaleOpacityRegularizeNoiseRelocationTrainer(model, scene_extent, *args, **kwargs),
        model, scene_extent, dataset,
        *args, **kwargs)


MCMCTrainer = ScaleOpacityRegularizeNoiseRelocationTrainer
CameraMCMCTrainer = CameraScaleOpacityRegularizeNoiseRelocationTrainer

# similar to BaseOpacityResetDensificationTrainer in gaussian_splatting.trainer.combinations


def NoiseRelocationTrainerWrapper(
        base_densifier_constructor: Callable[..., AbstractDensifier],
        model: GaussianModel, scene_extent: float,
        *args, **kwargs):
    return NoiseTrainerWrapper(
        partial(RelocationTrainerWrapper, base_densifier_constructor),
        model, scene_extent,
        *args, **kwargs)


def ScaleOpacityRegularizeNoiseRelocationTrainerWrapper(
        base_densifier_constructor: Callable[..., AbstractDensifier],
        model: GaussianModel, scene_extent: float,
        *args, **kwargs):
    return ScaleOpacityRegularizeTrainerWrapper(
        partial(NoiseRelocationTrainerWrapper, base_densifier_constructor),
        model, scene_extent,
        *args, **kwargs)


NoRegMCMCTrainerWrapper = NoiseRelocationTrainerWrapper
MCMCTrainerWrapper = ScaleOpacityRegularizeNoiseRelocationTrainerWrapper

# spacial, similar to DensificationTrainerWrapper in gaussian_splatting.trainer.densifier.combinations
