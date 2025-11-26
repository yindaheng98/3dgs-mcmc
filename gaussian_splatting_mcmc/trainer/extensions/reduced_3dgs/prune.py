from functools import partial
from typing import Callable
from gaussian_splatting import GaussianModel
from gaussian_splatting.dataset import CameraDataset, TrainableCameraDataset
from gaussian_splatting.trainer import DepthTrainerWrapper
from gaussian_splatting.trainer.densifier import AbstractDensifier, NoopDensifier
from gaussian_splatting_mcmc.trainer import MCMCTrainerWrapper
from reduced_3dgs.shculling import VariableSHGaussianModel, SHCullingTrainerWrapper
from reduced_3dgs import FullPruningDensifierWrapper


# Combinations of Relocation and Full Reduced Densifier

def MCMCFullPruningTrainerWrapper(
        base_densifier_constructor: Callable[..., AbstractDensifier],
        model: GaussianModel, scene_extent: float, dataset: CameraDataset,
        *args, **kwargs):
    return MCMCTrainerWrapper(
        partial(FullPruningDensifierWrapper, base_densifier_constructor),
        model, scene_extent, dataset,
        *args, **kwargs
    )


def BaseMCMCFullPruningTrainer(
        model: GaussianModel,
        scene_extent: float,
        dataset: CameraDataset,
        *args, **kwargs):
    return MCMCFullPruningTrainerWrapper(
        lambda model, *args, **kwargs: NoopDensifier(model),
        model, scene_extent, dataset,
        *args, **kwargs
    )


def DepthMCMCFullPruningTrainer(model: GaussianModel, scene_extent: float, dataset: TrainableCameraDataset, *args, **kwargs):
    return DepthTrainerWrapper(
        BaseMCMCFullPruningTrainer,
        model, scene_extent, dataset,
        *args, **kwargs
    )


MCMCFullPruningTrainer = DepthMCMCFullPruningTrainer


# Full Pruning Trainer + SH Culling

def SHCullingMCMCFullPruningTrainer(
    model: VariableSHGaussianModel,
        scene_extent: float,
        dataset: CameraDataset,
        *args, **kwargs):
    return SHCullingTrainerWrapper(
        MCMCFullPruningTrainer,
        model, scene_extent, dataset,
        *args, **kwargs
    )
