from functools import partial
from typing import Callable
from gaussian_splatting import GaussianModel
from gaussian_splatting.dataset import CameraDataset, TrainableCameraDataset
from gaussian_splatting.trainer import AbstractDensifier, CameraTrainerWrapper, DepthTrainerWrapper, CameraTrainerWrapper
from .noise import NoiseTrainerWrapper
from .relocate import BaseRelocationTrainer, RelocationTrainerWrapper
from .scale_opacity_reg import ScaleOpacityRegularizeTrainerWrapper

# similar to BaseDensificationTrainer in gaussian_splatting.trainer.densifier.combinations


def BaseNoiseRelocationTrainer(model: GaussianModel, dataset: CameraDataset, **configs):
    return NoiseTrainerWrapper(BaseRelocationTrainer, model, dataset, **configs)


def DepthNoiseRelocationTrainer(model: GaussianModel, dataset: CameraDataset, **configs):
    return DepthTrainerWrapper(BaseNoiseRelocationTrainer, model, dataset, **configs)


NoiseRelocationTrainer = DepthNoiseRelocationTrainer


def CameraNoiseRelocationTrainer(model: GaussianModel, dataset: TrainableCameraDataset, **configs):
    return CameraTrainerWrapper(
        lambda model, dataset, **configs: NoiseRelocationTrainer(model, dataset, **configs),
        model, dataset,
        **configs)


NoRegMCMCTrainer = NoiseRelocationTrainer
NoRegCameraMCMCTrainer = CameraNoiseRelocationTrainer


def BaseScaleOpacityRegularizeNoiseRelocationTrainer(model: GaussianModel, dataset: CameraDataset, **configs):
    return ScaleOpacityRegularizeTrainerWrapper(BaseNoiseRelocationTrainer, model, dataset, **configs)


def DepthScaleOpacityRegularizeNoiseRelocationTrainer(model: GaussianModel, dataset: CameraDataset, **configs):
    return DepthTrainerWrapper(BaseScaleOpacityRegularizeNoiseRelocationTrainer, model, dataset, **configs)


ScaleOpacityRegularizeNoiseRelocationTrainer = DepthScaleOpacityRegularizeNoiseRelocationTrainer


def CameraScaleOpacityRegularizeNoiseRelocationTrainer(model: GaussianModel, dataset: TrainableCameraDataset, **configs):
    return CameraTrainerWrapper(
        lambda model, dataset, **configs: ScaleOpacityRegularizeNoiseRelocationTrainer(model, dataset, **configs),
        model, dataset,
        **configs)


MCMCTrainer = ScaleOpacityRegularizeNoiseRelocationTrainer
CameraMCMCTrainer = CameraScaleOpacityRegularizeNoiseRelocationTrainer

# similar to BaseOpacityResetDensificationTrainer in gaussian_splatting.trainer.combinations


def NoiseRelocationTrainerWrapper(
        base_densifier_constructor: Callable[..., AbstractDensifier],
        model: GaussianModel, dataset: CameraDataset, *args,
        **configs):
    return NoiseTrainerWrapper(
        partial(RelocationTrainerWrapper, base_densifier_constructor),
        model, dataset, *args,
        **configs)


def ScaleOpacityRegularizeNoiseRelocationTrainerWrapper(
        base_densifier_constructor: Callable[..., AbstractDensifier],
        model: GaussianModel, dataset: CameraDataset, *args,
        **configs):
    return ScaleOpacityRegularizeTrainerWrapper(
        partial(NoiseRelocationTrainerWrapper, base_densifier_constructor),
        model, dataset, *args,
        **configs)


NoRegMCMCTrainerWrapper = NoiseRelocationTrainerWrapper
MCMCTrainerWrapper = ScaleOpacityRegularizeNoiseRelocationTrainerWrapper

# spacial, similar to DensificationTrainerWrapper in gaussian_splatting.trainer.densifier.combinations
