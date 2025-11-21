from typing import Callable
from gaussian_splatting import GaussianModel
from gaussian_splatting.dataset import TrainableCameraDataset
from gaussian_splatting.trainer import AbstractDensifier, CameraTrainerWrapper, DepthTrainerWrapper, CameraTrainerWrapper
from gaussian_splatting.trainer.densifier import NoopDensifier
from .noise import NoiseWrapper
from .relocate import Relocater, RelocationDensifierTrainerWrapper
from .scale_opacity_reg import ScaleOpacityRegularizeTrainerWrapper


def BaseRelocationTrainer(model: GaussianModel, scene_extent: float, *args, **kwargs):
    return RelocationDensifierTrainerWrapper(
        lambda model, scene_extent: NoopDensifier(model),
        model,
        scene_extent,
        *args, **kwargs
    )

# similar to BaseDensificationTrainer in gaussian_splatting.trainer.densifier.combinations


def NoiseRelocationTrainer(model: GaussianModel, scene_extent: float, *args, **kwargs):
    return NoiseWrapper(BaseRelocationTrainer, model, scene_extent, *args, **kwargs)


def DepthNoiseRelocationTrainer(model: GaussianModel, scene_extent: float, *args, **kwargs):
    return DepthTrainerWrapper(NoiseRelocationTrainer, model, scene_extent, *args, **kwargs)


def BaseScaleOpacityRegularizeTrainer(model: GaussianModel, scene_extent: float, *args, **kwargs):
    return ScaleOpacityRegularizeTrainerWrapper(NoiseRelocationTrainer, model, scene_extent, *args, **kwargs)


def DepthScaleOpacityRegularizeTrainer(model: GaussianModel, scene_extent: float, *args, **kwargs):
    return DepthTrainerWrapper(BaseScaleOpacityRegularizeTrainer, model, scene_extent, *args, **kwargs)


def CameraDepthNoiseRelocationTrainer(model: GaussianModel, scene_extent: float, dataset: TrainableCameraDataset, *args, **kwargs):
    return CameraTrainerWrapper(
        lambda model, scene_extent, dataset, *args, **kwargs: DepthNoiseRelocationTrainer(model, scene_extent, *args, **kwargs),
        model, scene_extent, dataset,
        *args, **kwargs)


def CameraDepthScaleOpacityRegularizeTrainer(model: GaussianModel, scene_extent: float, dataset: TrainableCameraDataset, *args, **kwargs):
    return CameraTrainerWrapper(
        lambda model, scene_extent, dataset, *args, **kwargs: DepthScaleOpacityRegularizeTrainer(model, scene_extent, *args, **kwargs),
        model, scene_extent, dataset,
        *args, **kwargs)


NoRegMCMCTrainer = DepthNoiseRelocationTrainer
MCMCTrainer = DepthScaleOpacityRegularizeTrainer
CameraNoRegMCMCTrainer = CameraDepthNoiseRelocationTrainer
CameraMCMCTrainer = CameraDepthScaleOpacityRegularizeTrainer

# similar to BaseOpacityResetDensificationTrainer in gaussian_splatting.trainer.combinations


def NoiseRelocationDensifierTrainerWrapper(
        noargs_base_densifier_constructor: Callable[[GaussianModel, float], AbstractDensifier],
        model: GaussianModel,
        scene_extent: float,
        *args,
        noise_lr=5e5,
        noise_from_iter=0,
        noise_until_iter=29_990,
        **kwargs):
    return NoiseWrapper(
        lambda model, scene_extent, *args, **kwargs: Relocater(model, scene_extent, noargs_base_densifier_constructor(model, scene_extent), *args, **kwargs),
        model, scene_extent,
        *args,
        noise_lr=noise_lr,
        noise_from_iter=noise_from_iter,
        noise_until_iter=noise_until_iter,
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

# spacial, similar to DensificationTrainerWrapper in gaussian_splatting.trainer.densifier.combinations
