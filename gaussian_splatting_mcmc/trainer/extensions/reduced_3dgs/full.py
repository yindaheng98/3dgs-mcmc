from functools import partial
from typing import Callable
from gaussian_splatting import GaussianModel, CameraTrainableGaussianModel
from gaussian_splatting.dataset import CameraDataset, TrainableCameraDataset
from gaussian_splatting.trainer import DepthTrainerWrapper, OpacityResetTrainerWrapper, CameraTrainerWrapper
from gaussian_splatting.trainer.densifier import AbstractDensifier, NoopDensifier
from gaussian_splatting_mcmc.trainer import MCMCTrainerWrapper
from reduced_3dgs.shculling import VariableSHGaussianModel, CameraTrainableVariableSHGaussianModel, SHCullingTrainerWrapper
from reduced_3dgs import FullReducedDensificationDensifierWrapper


# Combinations of MCMC and Full Reduced Densifier

def MCMCFullReducedDensificationTrainerWrapper(
        base_densifier_constructor: Callable[..., AbstractDensifier],
        model: GaussianModel, dataset: CameraDataset, *args,
        **configs):
    return MCMCTrainerWrapper(
        partial(FullReducedDensificationDensifierWrapper, base_densifier_constructor),
        model, dataset, *args,
        **configs
    )


def BaseMCMCFullReducedDensificationTrainer(
        model: GaussianModel,
        dataset: CameraDataset,
        **configs):
    return MCMCFullReducedDensificationTrainerWrapper(
        lambda model, dataset, **configs: NoopDensifier(model),
        model, dataset,
        **configs
    )


def DepthMCMCFullReducedDensificationTrainer(model: GaussianModel, dataset: TrainableCameraDataset, **configs):
    return DepthTrainerWrapper(
        BaseMCMCFullReducedDensificationTrainer,
        model, dataset,
        **configs
    )


MCMCFullReducedDensificationTrainer = DepthMCMCFullReducedDensificationTrainer


def CameraMCMCFullReducedDensificationTrainer(
        model: CameraTrainableGaussianModel,
        dataset: TrainableCameraDataset,
        **configs):
    return CameraTrainerWrapper(
        MCMCFullReducedDensificationTrainer,
        model, dataset,
        **configs
    )


# Full Reduced Densification Trainer + Opacity Reset

def OpacityResetMCMCFullReducedDensificationTrainer(
        model: GaussianModel,
        dataset: CameraDataset,
        **configs):
    return OpacityResetTrainerWrapper(
        MCMCFullReducedDensificationTrainer,
        model, dataset,
        **configs
    )


def CameraOpacityResetMCMCFullReducedDensificationTrainer(
        model: CameraTrainableGaussianModel,
        dataset: TrainableCameraDataset,
        **configs):
    return CameraTrainerWrapper(
        OpacityResetMCMCFullReducedDensificationTrainer,
        model, dataset,
        **configs
    )


# Full Reduced Densification Trainer + Opacity Reset + SH Culling

def SHCullingOpacityResetMCMCFullReducedDensificationTrainer(
        model: VariableSHGaussianModel,
        dataset: CameraDataset,
        **configs):
    return SHCullingTrainerWrapper(
        OpacityResetMCMCFullReducedDensificationTrainer,
        model, dataset,
        **configs
    )


def CameraSHCullingOpacityResetMCMCFullReducedDensificationTrainer(
        model: CameraTrainableVariableSHGaussianModel,
        dataset: TrainableCameraDataset,
        **configs):
    return CameraTrainerWrapper(
        SHCullingOpacityResetMCMCFullReducedDensificationTrainer,
        model, dataset,
        **configs
    )
