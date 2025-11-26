from functools import partial
from typing import Callable
from gaussian_splatting import GaussianModel, CameraTrainableGaussianModel
from gaussian_splatting.dataset import CameraDataset, TrainableCameraDataset
from gaussian_splatting.trainer import DepthTrainerWrapper, OpacityResetTrainerWrapper, CameraTrainerWrapper
from gaussian_splatting.trainer.densifier import AbstractDensifier, NoopDensifier
from gaussian_splatting_mcmc.trainer import MCMCTrainerWrapper
from reduced_3dgs.shculling import VariableSHGaussianModel, SHCullingTrainerWrapper
from reduced_3dgs import CameraTrainableVariableSHGaussianModel, FullReducedDensificationDensifierWrapper


# Combinations of MCMC and Full Reduced Densifier

def MCMCFullReducedDensificationTrainerWrapper(
        base_densifier_constructor: Callable[..., AbstractDensifier],
        model: GaussianModel, scene_extent: float, dataset: CameraDataset,
        *args, **kwargs):
    return MCMCTrainerWrapper(
        partial(FullReducedDensificationDensifierWrapper, base_densifier_constructor),
        model, scene_extent, dataset,
        *args, **kwargs
    )


def BaseMCMCFullReducedDensificationTrainer(
        model: GaussianModel,
        scene_extent: float,
        dataset: CameraDataset,
        *args, **kwargs):
    return MCMCFullReducedDensificationTrainerWrapper(
        lambda model, *args, **kwargs: NoopDensifier(model),
        model, scene_extent, dataset,
        *args, **kwargs
    )


def DepthMCMCFullReducedDensificationTrainer(model: GaussianModel, scene_extent: float, dataset: TrainableCameraDataset, *args, **kwargs):
    return DepthTrainerWrapper(
        BaseMCMCFullReducedDensificationTrainer,
        model, scene_extent, dataset,
        *args, **kwargs
    )


MCMCFullReducedDensificationTrainer = DepthMCMCFullReducedDensificationTrainer


def CameraMCMCFullReducedDensificationTrainer(
        model: CameraTrainableGaussianModel,
        scene_extent: float,
        dataset: TrainableCameraDataset,
        *args, **kwargs):
    return CameraTrainerWrapper(
        MCMCFullReducedDensificationTrainer,
        model, scene_extent, dataset,
        *args, **kwargs
    )


# Full Reduced Densification Trainer + Opacity Reset

def OpacityResetMCMCFullReducedDensificationTrainer(
        model: GaussianModel,
        scene_extent: float,
        dataset: CameraDataset,
        *args, **kwargs):
    return OpacityResetTrainerWrapper(
        MCMCFullReducedDensificationTrainer,
        model, scene_extent, dataset,
        *args, **kwargs
    )


def CameraOpacityResetMCMCFullReducedDensificationTrainer(
        model: CameraTrainableGaussianModel,
        scene_extent: float,
        dataset: TrainableCameraDataset,
        *args, **kwargs):
    return CameraTrainerWrapper(
        OpacityResetMCMCFullReducedDensificationTrainer,
        model, scene_extent, dataset,
        *args, **kwargs
    )


# Full Reduced Densification Trainer + Opacity Reset + SH Culling

def SHCullingOpacityResetMCMCFullReducedDensificationTrainer(
    model: VariableSHGaussianModel,
        scene_extent: float,
        dataset: CameraDataset,
        *args, **kwargs):
    return SHCullingTrainerWrapper(
        OpacityResetMCMCFullReducedDensificationTrainer,
        model, scene_extent, dataset,
        *args, **kwargs
    )


def CameraSHCullingOpacityResetMCMCFullReducedDensificationTrainer(
        model: CameraTrainableVariableSHGaussianModel,
        scene_extent: float,
        dataset: TrainableCameraDataset,
        *args, **kwargs):
    return CameraTrainerWrapper(
        SHCullingOpacityResetMCMCFullReducedDensificationTrainer,
        model, scene_extent, dataset,
        *args, **kwargs
    )
