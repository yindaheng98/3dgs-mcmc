from functools import partial
from typing import Callable
from gaussian_splatting import GaussianModel, CameraTrainableGaussianModel
from gaussian_splatting.dataset import CameraDataset, TrainableCameraDataset
from gaussian_splatting.trainer import DepthTrainerWrapper, CameraTrainerWrapper
from gaussian_splatting.trainer.densifier import AbstractDensifier, NoopDensifier
from gaussian_splatting_mcmc.trainer import MCMCTrainerWrapper
from reduced_3dgs.shculling import VariableSHGaussianModel, CameraTrainableVariableSHGaussianModel, SHCullingTrainerWrapper
from reduced_3dgs import FullPruningDensifierWrapper


# Combinations of MCMC and Full Pruning Densifier

def MCMCFullPruningTrainerWrapper(
        base_densifier_constructor: Callable[..., AbstractDensifier],
        model: GaussianModel, dataset: CameraDataset, *args,
        **configs):
    return MCMCTrainerWrapper(
        partial(FullPruningDensifierWrapper, base_densifier_constructor),
        model, dataset, *args,
        **configs
    )


def BaseMCMCFullPruningTrainer(
        model: GaussianModel,
        dataset: CameraDataset,
        **configs):
    return MCMCFullPruningTrainerWrapper(
        lambda model, dataset, **configs: NoopDensifier(model),
        model, dataset,
        **configs
    )


def DepthMCMCFullPruningTrainer(model: GaussianModel, dataset: TrainableCameraDataset, **configs):
    return DepthTrainerWrapper(
        BaseMCMCFullPruningTrainer,
        model, dataset,
        **configs
    )


MCMCFullPruningTrainer = DepthMCMCFullPruningTrainer


def CameraMCMCFullPruningTrainer(
        model: CameraTrainableGaussianModel,
        dataset: TrainableCameraDataset,
        **configs):
    return CameraTrainerWrapper(
        MCMCFullPruningTrainer,
        model, dataset,
        **configs
    )


# Full Pruning Trainer + SH Culling

def SHCullingMCMCFullPruningTrainer(
        model: VariableSHGaussianModel,
        dataset: CameraDataset,
        **configs):
    return SHCullingTrainerWrapper(
        MCMCFullPruningTrainer,
        model, dataset,
        **configs
    )


def CameraSHCullingMCMCFullPruningTrainer(
        model: CameraTrainableVariableSHGaussianModel,
        dataset: TrainableCameraDataset,
        **configs):
    return CameraTrainerWrapper(
        SHCullingMCMCFullPruningTrainer,
        model, dataset,
        **configs
    )
